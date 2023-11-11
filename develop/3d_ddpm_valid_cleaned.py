if not os.path.exists('./generated_images/'): os.makedirs('./generated_images/')
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
import SimpleITK as sitk
import torchio as tio
import torch.nn as nn
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager
import nibabel as nib
from typing import Optional
from nilearn.plotting import plot_anat
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')
JUPYTER_ALLOW_INSECURE_WRITES=True


# ---------------------------------- #

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,5,6'
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the default number of threads for SimpleITK.
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
# Set the global behavior of SimpleITK to use the Platform multithreading.
# This is especially useful when using SimpleITK with multi-core systems.
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")


config = {
    'batch_size': 4,
    'imgDimResize':(160,192,160),
    'imgDimPad': (208, 256, 208),
    'spatialDims': '3D',
    'unisotropic_sampling': True, 
    'perc_low': 1, 
    'perc_high': 99,
    'rescaleFactor':2,
    'base_path': '/scratch1/akrami/Latest_Data/Data',
}



# ---------------------------------- #

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
        
        subject['vol'].data = subject['vol'].data.permute(0,3, 2, 1)
 
        return subject

def get_augment(cfg): # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment



# ---------------------------------- #

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


def Eval(csv,cfg): 
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path,reader=sitk_reader).shape != tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape:
            print(f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path,reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')
        

        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path, reader=sitk_reader), 
            'seg' : tio.ScalarImage(sub.seg_path, reader=sitk_reader),
            'age' : sub.age,
            'ID' : sub.img_name,
            'path' : sub.img_path
        }
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
    return ds






# ---------------------------------- #

imgpath = {}
csvpath_train = '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/IXI_train_fold0.csv'
pathBase = '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data_train'
csvpath_val = '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/IXI_val_fold0.csv'
csvpath_test = '/acmenas/hakrami/patched-Diffusion-Models-UAD/Data/splits/Brats21_sub_test.csv'
var_csv = {}
states = ['train','val','test']
var_csv['train'] = pd.read_csv(csvpath_train)
var_csv['val'] = pd.read_csv(csvpath_val)
var_csv['test'] = pd.read_csv(csvpath_test)

for state in states:
    var_csv[state]['settype'] = state
    var_csv[state]['img_path'] = pathBase  + var_csv[state]['img_path']
    var_csv[state]['mask_path'] = pathBase  + var_csv[state]['mask_path']
    if state!='test':
        var_csv[state]['seg_path'] = None

var_csv['test']['seg_path'] =  pathBase  + var_csv['test']['seg_path']
    
data_train = Train(var_csv['train'],config) 
data_val = Train(var_csv['val'],config)                
data_test = Eval(var_csv['test'],config)



#data_train = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_train_fold0.csv', converters={'img_path': pd.eval}), config)
train_loader = DataLoader(data_train, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_val = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/IXI_val_fold0.csv', converters={'img_path': pd.eval}), config)
val_loader = DataLoader(data_val, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)

#data_test = Train(pd.read_csv('/project/ajoshi_27/akrami/monai3D/GenerativeModels/data/split/Brats21_test.csv', converters={'img_path': pd.eval}), config)
test_loader = DataLoader(data_test, batch_size=config.get('batch_size', 1),shuffle=True,num_workers=8)


# ---------------------------------- #

s = next(iter(test_loader))
print(s.keys())
print(s['age'])
print(s['vol']['data'].size())
image_array = s['vol']['data']
plt.imshow(image_array[0][0][:,:,40].squeeze())
plt.savefig(os.path.join("./generated_images/", "image_org" + ".png"))
plt.close()
# ---------------------------------- #


print(s['seg']['data'].size())
image_array = s['seg']['data'][0][0][:,:,40].squeeze().cpu().numpy()
plt.imshow(image_array, cmap='gray')
plt.savefig(os.path.join("./generated_images/", "image_seg" + ".png"))
plt.close()

# ---------------------------------- #

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
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)

inferer = DiffusionInferer(scheduler)

optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

# ---------------------------------- #

# specify your model filename
model_filename = '/acmenas/hakrami/3D_lesion_DF/models/model_epoch424.pt'

# load state_dict into the model
model.load_state_dict(torch.load(model_filename))
model.eval()

# ---------------------------------- #


noise = torch.randn((1, 1, 80, 96, 80))
noise = noise.to(device)
scheduler.set_timesteps(num_inference_steps=1000)
image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)

# ---------------------------------- #

plt.style.use("default")
plotting_image_0 = np.concatenate([image[0, 0, :, :, 15].cpu(), np.flipud(image[0, 0, :, 20, :].cpu().T)], axis=1)
plotting_image_1 = np.concatenate([np.flipud(image[0, 0, 15, :, :].cpu().T), np.zeros((80, 80))], axis=1)
plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0))
plt.savefig(os.path.join("./generated_images/", "image_gen" + ".png"))
plt.close()
# ---------------------------------- #



image_data = image.squeeze().cpu().numpy()

# Change datatype to float64 (optional, only if your data is not complex)
image_data = image_data.astype(np.float64)
nii_img = nib.Nifti1Image(image_data, np.eye(4))

# Save the image
nib.save(nii_img, 'test_image.nii')

# ---------------------------------- #

def print_nifti_shape(file_path):
    try:
        # Load the NIfTI files
        nifti_img = nib.load(file_path)

        # Get the shape of the NIfTI data array
        data_shape = nifti_img.get_fdata().shape

        # Print the shape
        print(f"Shape of the NIfTI file '{file_path}': {data_shape}")
    except Exception as e:
        print(f"Error: {e}")



display = plotting.plot_anat("test_image.nii")
display.savefig("output_filename.png")
plotting.show()
# Replace 'your_nifti_file.nii' with the actual path to your NIfTI file
nifti_file_path = "test_image.nii"
print_nifti_shape(nifti_file_path)


# ---------------------------------- # denoise one sample

noise =torch.randn_like(image_array)
noisy_img_v2 = scheduler.add_noise(original_samples=image_array, noise=noise, timesteps=torch.tensor(500))
noise = noisy_img_v2[0:1,:,:,:,:].to(device)
scheduler.set_timesteps(num_inference_steps=1000)
image_xx = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
diff_image = torch.abs(image_xx.to(device)-image_array[0:1,:,:,:,:].to(device))


# Compute the global min and max values
global_min = min(noisy_img_v2.min().item(), image_xx.min().item(), diff_image.min().item())
global_max = max(noisy_img_v2.max().item(), image_xx.max().item(), diff_image.max().item())

# Noisy Image
plt.imshow(noisy_img_v2[0][0][:,:,40].squeeze(), vmin=global_min, vmax=global_max)
plt.savefig(os.path.join("./generated_images/", "image_noisy"+".png"))
plt.close()

# Denoised Image
plt.imshow(image_xx[0][0][:,:,40].squeeze(), vmin=global_min, vmax=global_max)
plt.savefig(os.path.join("./generated_images/", "image_denoised"+ ".png"))
plt.close()

# Difference Image
plt.imshow(diff_image[0][0][:,:,40].squeeze(), vmin=global_min, vmax=global_max)
plt.savefig(os.path.join("./generated_images/", "image_diff" + ".png"))
plt.close()

# ---------------------------------- # save nifti

image_data = image_xx.squeeze().cpu().numpy()
image_data = image_data.astype(np.float64)
nii_img = nib.Nifti1Image(image_data, np.eye(4))
nib.save(nii_img, 'test_brats_recon_image.nii')
image_data = image_array.squeeze().cpu().numpy()
image_data = image_data.astype(np.float64)
nii_img = nib.Nifti1Image(image_data, np.eye(4))
nib.save(nii_img, 'test_brats_image.nii')

