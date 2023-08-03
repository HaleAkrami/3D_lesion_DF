import os
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import AddChanneld, CenterSpatialCropd, Compose, Lambdad, LoadImaged, Resized, ScaleIntensityd
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import scipy
import io
import random
import math
import pandas as pd
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
import numpy as np
import SimpleITK as sitk
import torchio as tio
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager

from typing import Optional

import warnings
warnings.filterwarnings('ignore')

# Set the manualSeed, random seed, and device
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# It's recommended to set the global deterministic behavior of some libraries.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the default number of threads for SimpleITK.
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

# Set the global behavior of SimpleITK to use the Platform multithreading.
# This is especially useful when using SimpleITK with multi-core systems.
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")



# Replace 'path_to_your_nii_file.nii' with the actual path to your NIfTI image file


config = {
    'batch_size': 2,
    'imgDimResize':(160,192,160),
    'imgDimPad': (208, 256, 208),
    'spatialDims': '3D',
    'unisotropic_sampling': True, 
    'perc_low': 1, 
    'perc_high': 99,
    'rescaleFactor':2,
    'base_path': '/scratch1/akrami/Latest_Data/Data',
}


def Train(csv,cfg,preload=True):
    subjects = []
    base_path = cfg.get('base_path',None)
    h, w, d = tuple(cfg.get('imgDimResize',(160,192,160)))
    for _, sub in csv.iterrows():
       # print(sub.img_path)
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path, reader=sitk_reader), 
            'age' : sub.age,
            'ID' : sub.img_name,
            'path' : sub.img_path
        }

        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)
        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    
    if preload: 
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment = get_augment(cfg))
    else: 
        ds = tio.SubjectsDataset(subjects, transform = tio.Compose([get_transform(cfg),get_augment(cfg)]))
        
    if cfg.get('spatialDims') == '2D':
        slice_ind = cfg.get('startslice',None) 
        seq_slices = cfg.get('sequentialslices',None) 
        ds = vol2slice(ds,cfg,slice=slice_ind,seq_slices=seq_slices)
    return ds

def sitk_reader(path):
                
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2,1,0)
    return vol, None


def get_transform(cfg): # only transforms that are applied once before preloading
    exclude_from_resampling = None
    h, w, d = tuple(cfg.get('imgDimResize',(160,192,160)))
    if cfg.get('unisotropic_sampling',True):
        preprocess = tio.Compose([
        tio.CropOrPad((h,w,d),padding_mode=0),
        tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99))),
                tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
            ])


    return preprocess


class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)

class preload_wrapper(Dataset):
    def __init__(self,ds,cache,augment=None):
            self.cache = cache
            self.ds = ds
            self.augment = augment
    def reset_memory(self):
        self.cache.reset()
    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        if self.cache.is_cached(index) :
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject
    
class vol2slice(Dataset):
    def __init__(self,ds,cfg,onlyBrain=False,slice=None,seq_slices=None):
            self.ds = ds
            self.onlyBrain = onlyBrain
            self.slice = slice
            self.seq_slices = seq_slices
            self.counter = 0 
            self.ind = None
            self.cfg = cfg

    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        
        #Commenting the next few lines to introduce a simple masking operation
#         if self.onlyBrain:
#             start_ind = None
#             for i in range(subject['vol'].data.shape[-1]):
#                 if subject['mask'].data[0,:,:,i].any() and start_ind is None: # only do this once
#                     start_ind = i 
#                 if not subject['mask'].data[0,:,:,i].any() and start_ind is not None: # only do this when start_ind is set
#                     stop_ind = i 
#             low = start_ind
#             high = stop_ind
#         else: 
#             low = 0
#             high = subject['vol'].data.shape[-1]
#         if self.slice is not None:
#             self.ind = self.slice
#             if self.seq_slices is not None:
#                 low = self.ind
#                 high = self.ind + self.seq_slices
#                 self.ind = torch.randint(low,high,size=[1])
#         else:
#             if self.cfg.get('unique_slice',False): # if all slices in one batch need to be at the same location
#                 if self.counter % self.cfg.batch_size == 0 or self.ind is None: # only change the index when changing to new batch
#                     self.ind = torch.randint(low,high,size=[1])
#                 self.counter = self.counter +1
#             else: 
#                 self.ind = torch.randint(low,high,size=[1])

#         subject['ind'] = self.ind

#         subject['vol'].data = subject['vol'].data[...,self.ind]
#         subject['mask'].data = subject['mask'].data[...,self.ind]


        subject['vol'].data = subject['vol'].data.permute(0,3, 2, 1)
    
#         #msk_normal = ~np.all(subject['vol'].data.numpy()[0] == 0,axis=(1,2)) # Remove empty planes (Replacing this line of code to include slices that have significant prain volume in them (more than 15% of the image is brain))
        
#         msk_normal = (np.count_nonzero(subject['vol'].data.numpy()[0],axis=(1,2))/(subject['vol'].data.numpy()[0].shape[1]*subject['vol'].data.numpy()[0].shape[2]))>=0.15
        
#         choices = np.arange(len(msk_normal))[msk_normal]
        
#         sample_idx = np.array(random.choices(choices,k = 1))
        
#         subject['vol'].data = subject['vol'].data[0:2, sample_idx, ...]

        return subject

def get_augment(cfg): # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment



imgpath = {}
csvpath_train = '/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_train_fold0.csv'
pathBase = '/scratch1/akrami/Data_train'
csvpath_val = '/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_val_fold0.csv'
csvpath_test = '/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/Brats21_test.csv'
var_csv = {}
states = ['train','val','test']
var_csv['train'] = pd.read_csv(csvpath_train)
var_csv['val'] = pd.read_csv(csvpath_val)
var_csv['test'] = pd.read_csv(csvpath_test)
# if cfg.mode == 't2':
#     keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) # only keep t2 images that have a t1 counterpart

for state in states:
    var_csv[state]['settype'] = state
    var_csv[state]['img_path'] = pathBase  + var_csv[state]['img_path']
    var_csv[state]['mask_path'] = pathBase  + var_csv[state]['mask_path']
    var_csv[state]['seg_path'] = None

    # if cfg.mode == 't2': 
    #     var_csv[state] =var_csv[state][var_csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
    #     var_csv[state]['img_path'] = var_csv[state]['img_path'].str.replace('t1','t2')
    
    
data_train = Train(var_csv['train'],config) 
data_val = Train(var_csv['val'],config)                
data_test = Train(var_csv['test'],config)



#data_train = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_train_fold0.csv', converters={'img_path': pd.eval}), config)
train_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=1)

#data_val = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_val_fold0.csv', converters={'img_path': pd.eval}), config)
val_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=1)

#data_test = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/Brats21_test.csv', converters={'img_path': pd.eval}), config)
test_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=1)


device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=[256, 256, 512],
    attention_levels=[False, False, True],
    num_head_channels=[0, 0, 512],
    num_res_blocks=2,
)
model.to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)

inferer = DiffusionInferer(scheduler)

optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)


n_epochs = 500
val_interval = 25
epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
       # images = batch["image"].to(device)
        images = batch['vol']['data'].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batchbatch['vol']['data'].to(device)
            noise = torch.randn_like(images).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        #80, 96, 80
        image = torch.randn((1, 1, 80, 96, 80))
        image = image.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)

        plt.figure(figsize=(2, 2))
        plt.imshow(image[0, 0, :, :, 15].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        # Modify the filename to include the epoch number
        filename = f"./results/sample_epoch{epoch}.png"

        plt.savefig(filename, dpi=300)  
        # Save the model
        model_filename = f"./models/model_epoch{epoch_number}.pt"
        torch.save(model.state_dict(), model_filename)

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")