# %% [markdown]
# # Imports

# %%
# !python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# %%
# Standard Libraries
import os
import io
import math
import time
import random
import tempfile
import warnings
from multiprocessing import Manager
from typing import Optional
import tqdm
# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
# MONAI Libraries
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism

# Custom Libraries
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from dataloader import Train ,Eval 

# Configuration
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
warnings.filterwarnings('ignore')
import wandb
wandb.init(project='3D_ddpm',name='test')

JUPYTER_ALLOW_INSECURE_WRITES=True

# %% [markdown]
# # Set seeds and configs

# %%
# Initialize Configuration
config = {
    'batch_size': 1,
    'imgDimResize': (160, 192, 160),
    'imgDimPad': (208, 256, 208),
    'spatialDims': '3D',
    'unisotropic_sampling': True,
    'perc_low': 1,
    'perc_high': 99,
    'rescaleFactor': 2,
    'base_path': '/scratch1/akrami/Latest_Data/Data',
}

# Seed and Device Configuration
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA and CUDNN Configuration
# Uncomment the following line to specify CUDA_VISIBLE_DEVICES
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# SimpleITK Configuration
# Set the default number of threads and global behavior for SimpleITK
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")



# %% [markdown]
# # Load the data

# %%
imgpath = {}
# '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv'
#'/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/IXI_train_fold0.csv',
#csvpath_trains = ['/project/ajoshi_27/akrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv', '/project/ajoshi_27/akrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv']
csvpath_trains=['./Data/splits/combined_4datasets.csv']
pathBase = '/scratch1/akrami/Data_train'
csvpath_val = './Data/splits/IXI_val_fold0.csv'
csvpath_test = './Data/splits/Brats21_sub_test.csv'
var_csv = {}
states = ['train','val','test']

df_list = []

# Loop through each CSV file path and read it into a DataFrame
for csvpath in csvpath_trains:
    df = pd.read_csv(csvpath)
    df_list.append(df)
# %%


var_csv['train'] =pd.concat(df_list, ignore_index=True)
var_csv['val'] = pd.read_csv(csvpath_val)
var_csv['test'] = pd.read_csv(csvpath_test)
# if cfg.mode == 't2':
#     keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) # only keep t2 images that have a t1 counterpart

for state in states:
    var_csv[state]['settype'] = state
    var_csv[state]['norm_path'] = ''
    var_csv[state]['img_path'] = pathBase  + var_csv[state]['img_path']
    var_csv[state]['mask_path'] = pathBase  + var_csv[state]['mask_path']
    if state != 'test':
        var_csv[state]['seg_path'] = None
    else:
        var_csv[state]['seg_path'] = pathBase  + var_csv[state]['seg_path']

    # if cfg.mode == 't2': 
    #     var_csv[state] =var_csv[state][var_csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
    #     var_csv[state]['img_path'] = var_csv[state]['img_path'].str.replace('t1','t2')
    
    
data_train = Train(var_csv['train'],config) 
data_val = Train(var_csv['val'],config)                
data_test = Eval(var_csv['test'],config)


#data_train = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_train_fold0.csv', converters={'img_path': pd.eval}), config)
#train_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_val = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_val_fold0.csv', converters={'img_path': pd.eval}), config)
val_loader = DataLoader(data_val, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_test = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/Brats21_test.csv', converters={'img_path': pd.eval}), config)
test_loader = DataLoader(data_test, batch_size=1,shuffle=False,num_workers=8)


device = torch.device("cuda")

# %% [markdown]
# # Load the model

# %%
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=[32, 64, 64],
    attention_levels=[False, False,True],
    num_head_channels=[0, 0,32],
    num_res_blocks=2,
)
model.to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)

inferer = DiffusionInferer(scheduler)

optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

# %%
# specify your model filename
#model_filename = '/scratch1/akrami/models/3Ddiffusion/half/model_epoch984.pt'
model_filename ='/scratch1/akrami/storage/DF_results/models/model_large_epoch999.pt'
# load state_dict into the model
model.load_state_dict(torch.load(model_filename))

# if you need to set the model in evaluation mode
model.eval()

# %% [markdown]
# # Generate an Image

# %%
def denoise(noised_img,sample_time,scheduler,inferer,model):
    with torch.no_grad():
        with autocast(enabled=True):
            for t in range(sample_time - 1, -1, -1):
                batch_size = noised_img.size(0)  # Get the batch size
                t_batch=torch.Tensor((t,)).to(noised_img.device)
                t_batch = t_batch.unsqueeze(0).expand(batch_size, -1)  # Expand tensor `t` to have the desired batch size
                t_batch = t_batch.to(noised_img.device)[:,0] 
                model_output = model(noised_img, timesteps=t_batch)
                noised_img, _ = scheduler.step(model_output, t, noised_img)
            return noised_img

    

# %%

# %%
sample_time = 500
i = 0
all_errors = []


model.eval()
progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), ncols=70)
for step, batch in progress_bar:
    images = batch['vol']['data'].to(device)
    # Expand the dimensions of batch['peak'] to make it [1, 1, 1, 1, 4]
    peak_expanded = (batch['peak'].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).long()
    peak_expanded = peak_expanded.to(device)

    # Perform the division
    images = (images / peak_expanded)
    
    middle_slice_idx = images.size(-1) // 2  # Define middle_slice_idx here

    noise = torch.randn_like(images)
    noisy_img = scheduler.add_noise(original_samples=images, noise=noise, timesteps=torch.tensor(sample_time))
    noisy_img = noisy_img.to(device)
    denoised_sample = denoise(noisy_img, sample_time, scheduler, inferer, model)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(images[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(noisy_img[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    
    axes[1, 0].imshow(denoised_sample[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
    axes[1, 0].set_title('Denoised Image')

    error = torch.abs(images - denoised_sample)
    axes[1, 1].imshow(error[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
    axes[1, 1].set_title('Error Image')
    
    plt.tight_layout()
    plt.show()

    wandb.log({"sample_image_val": [wandb.Image(plt)]})
    plt.close()  # Close the figure to free up memory

    all_errors.append(error.flatten())


# Stack all error values to form a big tensor
all_errors_tensor = torch.cat(all_errors)



# print(all_errors_tensor.shape)


# %%
all_errors_numpy = all_errors_tensor.cpu().numpy()
threshold = np.quantile(all_errors_numpy, 0.95)
print(f"Only 5% of the error values exceed: {threshold}")


# %%
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import MaskedLoss
from monai.losses import MaskedDiceLoss
# ssim_loss = MaskedLoss(SSIMLoss,spatial_dims=3)
dice_loss = MaskedDiceLoss()
ssim_loss = MaskedLoss(SSIMLoss,spatial_dims=3, data_range=1)




# %%
# =================== Test Loop for Dice Calculation ===================

# =================== Test Loop for Dice Calculation ===================

# def dice_coefficient(prediction, target):
#     """Compute the Dice coefficient only for non-background region."""
#     smooth = 1.0
#     mask = target > 0  # mask of non-zero (non-background) values
    
#     intersection = (prediction * target * mask).sum()
#     return (2. * intersection + smooth) / ((prediction * mask).sum() + (target * mask).sum() + smooth)


all_dices = []
all_ssim_values=[]

for step, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
    images = batch['vol']['data'].to(device)
    # Expand the dimensions of batch['peak'] to make it [1, 1, 1, 1, 4]
    peak_expanded = (batch['peak'].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).long()
    peak_expanded = peak_expanded.to(device)
    images = (images / peak_expanded)
    middle_slice_idx = images.size(-1) // 2  # Define middle_slice_idx here


    
    data_range = images.max()
    print(data_range)
    ssim_loss = MaskedLoss(SSIMLoss,spatial_dims=3, data_range=data_range)
    
    

    noise = torch.randn_like(images)
    noisy_img = scheduler.add_noise(original_samples=images, noise=noise, timesteps=torch.tensor(sample_time))
    noisy_img = noisy_img.to(device)
    denoised_sample = denoise(noisy_img, sample_time, scheduler, inferer, model)

    error = torch.abs(images - denoised_sample)
    thresholded_error = (error > threshold).float().to(device)
    gt_segmentation = (batch['seg']['data']>0).float().to(device)
    
  
    non_background_mask = (images > 0).float()
    gt_segmentation = batch['seg']['data'].to(device)
    non_segmented_mask = (gt_segmentation == 0).float()
    combined_mask = non_background_mask * non_segmented_mask

    # Apply the combined mask to the original and denoised images
    # masked_original = images 
    # masked_denoised = denoised_sample * combined_mask
    
    # Calculate Dice score
    dice_score = 1-dice_loss(thresholded_error, gt_segmentation,non_background_mask)
    all_dices.append(dice_score.item())
    # Calculate SSIM for the regions specified by the combined mask
    ssim_val =1- ssim_loss(images, denoised_sample,combined_mask)
    all_ssim_values.append(ssim_val.item())


    if step % 10 ==0:
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        # Original Image
        axes[0, 0].imshow(images[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        # Segmented Image
        image_array_slice = batch['seg']['data'][i][0][:,:,middle_slice_idx].squeeze().cpu().numpy()
        axes[0, 1].imshow(image_array_slice, cmap='gray')
        axes[0, 1].set_title('Segmented Image')
        
        # Transformed Image
        axes[1, 0].imshow(denoised_sample[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=2, cmap='gray')
        axes[1, 0].set_title('Transformed Image')
        
        # Thresholded Difference Image
        
        axes[1, 1].imshow(thresholded_error[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=1, cmap='gray')
        axes[1, 1].set_title('Thresholded Difference Image')

        axes[1, 2].imshow(error[i][0][:,:,middle_slice_idx].squeeze().cpu().numpy(), vmin=0, vmax=1, cmap='gray')
        axes[1, 1].set_title('error image')
        
        # Save the entire figure
        plt.tight_layout()
        
         
        plt.show()
        wandb.log({"sample_image_test": [wandb.Image(plt)]})

    
# Average Dice score
avg_dice_score = sum(all_dices) / len(all_dices)
print(f"Average Dice score over the test set: {avg_dice_score}")

# Average SSIM values
avg_ssim_value = sum(all_ssim_values) / len(all_ssim_values)
print(f"Average SSIM between specified regions of original and denoised images: {avg_ssim_value}")



# # %%


# # %%



