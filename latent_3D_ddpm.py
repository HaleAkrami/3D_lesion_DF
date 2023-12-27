# %%
import wandb
wandb.init(project='latent_ddpm_3D',name='small_3D')
import wandb


# Standard libraries
import os
import tempfile
import time
import io
import random
import math
import warnings
from multiprocessing import Manager
from typing import Optional
from torch.nn import L1Loss
# Data manipulation libraries
import numpy as np
import pandas as pd
import scipy

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

# MONAI libraries
# from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    AddChanneld, 
    CenterSpatialCropd, 
    Compose, 
    Lambdad, 
    LoadImaged, 
    Resized, 
    ScaleIntensityd
)
from monai.utils import set_determinism
from monai.utils import first, set_determinism
# Other medical image processing libraries
import SimpleITK as sitk
import torchio as tio

# Plotting and visualization
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# Custom modules
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet,AutoencoderKL
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

# Weights and Biases for experiment tracking
from dataloader import Train,Eval
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6,7"





config = {
    'batch_size': 32,
    'imgDimResize':(160,192,160),
    'imgDimPad': (208, 256, 208),
    'spatialDims': '3D',
    'unisotropic_sampling': True, 
    'perc_low': 0, 
    'perc_high': 100,
    'rescaleFactor':2,
    'base_path': '/scratch1/akrami/Latest_Data/Data',
}

# %%
wandb.config.update(config )


imgpath = {}
# '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv'
#'/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/IXI_train_fold0.csv',
#csvpath_trains = ['/project/ajoshi_27/akrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv', '/project/ajoshi_27/akrami/patched-Diffusion-Models-UAD/Data/splits/BioBank_train.csv']
csvpath_trains=['/acmenas/hakrami/3D_lesion_DF/Data/splits/combined_4datasets.csv']
pathBase = '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data_train'
csvpath_val = '/acmenas/hakrami/3D_lesion_DF/Data/splits/IXI_val_fold0.csv'
csvpath_test = '/acmenas/hakrami/3D_lesion_DF/Data/splits/Brats21_sub_test.csv'
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
train_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_val = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_val_fold0.csv', converters={'img_path': pd.eval}), config)
val_loader = DataLoader(data_val, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_test = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/Brats21_test.csv', converters={'img_path': pd.eval}), config)
test_loader = DataLoader(data_test, batch_size=config.get('batch_size', 1),shuffle=False,num_workers=16)


device = torch.device("cuda")

autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)


check_data = first(train_loader)
images = check_data['vol']['data'].to(device)
# Expand the dimensions of sub_test['peak'] to make it [1, 1, 1, 1, 4]
peak_expanded = (check_data['peak'].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).long()
# Move both tensors to the device
peak_expanded = peak_expanded.to(device)

# Perform the division
images = (images / peak_expanded)

autoencoder.to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    autoencoder = nn.DataParallel(autoencoder)

 
model_filename = '/acmenas/hakrami/3D_lesion_DF/models/latent_3D/model_KL_epoch449.pt'
autoencoder.load_state_dict(torch.load(model_filename))


unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=3,
    num_res_blocks=1,
    num_channels=(32, 64, 64),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 64),
)
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.module.encode_stage_2_inputs(images.to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)


wandb.watch(unet, log_freq=100)


n_epochs = 1000
epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()

z = autoencoder.module.encode_stage_2_inputs(images.to(device))
val_interval =25

for epoch in range(n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        optimizer_diff.zero_grad(set_to_none=True)
       # images = batch["image"].to(device)
        images = batch['vol']['data'].to(device)
        images[images<0.01]=0
        # Expand the dimensions of sub_test['peak'] to make it [1, 1, 1, 1, 4]
        peak_expanded = (batch['peak'].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).long()
        # Move both tensors to the device
        peak_expanded = peak_expanded.to(device)
        #z = autoencoder.module.encode_stage_2_inputs(images.to(device))
        # Perform the division
        images = (images / peak_expanded)
        batch_size = images.size(0)  # 0 for the first dimension, which is batch size



        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn(batch_size, *z.size()[1:]).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))
    wandb.log({"loss_train": epoch_loss / (step + 1)})
    

    if (epoch) % val_interval == 0:
        epoch_loss_val =0
        unet.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch['vol']['data'].to(device)
            images[images<0.01]=0
            peak_expanded = (batch['peak'].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).long()
            # Move both tensors to the device
            peak_expanded = peak_expanded.to(device)
            batch_size = images.size(0)
            # Perform the division
            images = (images / peak_expanded)
            noise = torch.randn(batch_size, *z.size()[1:]).to(device)
            #z = autoencoder.module.encode_stage_2_inputs(images.to(device))
            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

            
            epoch_loss_val += loss.item()
        wandb.log({"loss_val": epoch_loss_val / (step + 1)})
        noise = torch.randn_like(z).to(device) [0:1,:,:,:,:]
        scheduler.set_timesteps(num_inference_steps=1000)
        synthetic_images = inferer.sample(input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler)
        middle_slice_idx = images.size(-1) // 2
        plt.figure(figsize=(2, 2))
        plt.imshow(synthetic_images[0, 0, :, :, middle_slice_idx].cpu().detach().numpy(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        wandb.log({"sample_image": [wandb.Image(plt)]})

        model_filename = f"./models/norm3/latent_KL_epoch{epoch}.pt"
        torch.save(unet.state_dict(), model_filename)


torch.cuda.empty_cache()


