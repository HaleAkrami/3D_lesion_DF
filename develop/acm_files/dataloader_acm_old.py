import wandb
wandb.init(project='33_ddpm')


# Standard libraries
import os
import monai
import tempfile
import time
import io
import random
import math
import warnings
from multiprocessing import Manager
from typing import Optional

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
from monai.apps import DecathlonDataset
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

# Other medical image processing libraries
import SimpleITK as sitk
import torchio as tio

# Plotting and visualization
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# Custom modules
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


import torch
import torchio as tio
from torchio import Image, DATA




def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def apply_hist_norm(image):
    hist_norm = HistogramNormalize(num_bins=256, min=0, max=1)
    return hist_norm(image)


def normalize_by_peak(image):
    # Calculate histogram, ignoring first bin
    image[image<0] =0
    hist, bins = np.histogram(image, bins=20)
    
    # Find the peak of the histogram
    peak = bins[np.argmax(hist[1:])+1]  # Adding 1 to exclude the first bin
    #print(peak)
    
    # Normalize the image by the peak
    normalized_image = image / peak
    
    return normalized_image, peak

def Train(csv,cfg,preload=True):
    subjects = []
    base_path = cfg.get('base_path',None)
    h, w, d = tuple(cfg.get('imgDimResize',(160,192,160)))
    for _, sub in csv.iterrows():
       # print(sub.img_path)
        vol_image = tio.ScalarImage(sub.img_path, reader=sitk_reader)
        # vol_array = vol_image.numpy()
        
        # # Apply custom normalization
        # normalized_vol_array, peak = normalize_by_peak(vol_array)
        # print(peak)
        
        # # Get spacing information
        # spacing = vol_image.spacing

        # # Convert back to ScalarImage
        # normalized_vol_image = tio.ScalarImage(tensor=normalized_vol_array, spacing=spacing)

        
        # Apply histogram normalization here
        #vol_image = apply_hist_norm(vol_image)
        subject_dict = {
            'vol': vol_image,
            'peak': sub.peak,  # Store the peak value
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


        vol_image = tio.ScalarImage(sub.img_path, reader=sitk_reader)
        # vol_array = vol_image.numpy()
        
        # # Apply custom normalization
        # normalized_vol_array, peak = normalize_by_peak(vol_array)
        
        # # Get spacing information
        # spacing = vol_image.spacing

        # # Convert back to ScalarImage
        # normalized_vol_image = tio.ScalarImage(tensor=normalized_vol_array, spacing=spacing)

        
        # Apply histogram normalization here
        #vol_image = apply_hist_norm(vol_image)
        subject_dict = {
            'vol': vol_image,
            'peak': sub.peak,  # Store the peak value
            'seg' : tio.LabelMap(sub.seg_path, reader=sitk_reader, type=tio.LABEL),
            'age' : sub.age,
            'ID' : sub.img_name,
            'path' : sub.img_path
        }
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform = get_eval_transform(cfg))
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
        #tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling)
        #,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        print('dummy')


    return preprocess

def apply_hist_norm(x):
    hist_norm = monai.transforms.HistogramNormalize(num_bins=256, min=0, max=1)
    return hist_norm(x)

def get_transform_v2(cfg):
    exclude_from_resampling = None
    h, w, d = tuple(cfg.get('imgDimResize', (160, 192, 160)))
    
    if cfg.get('unisotropic_sampling', True):
        preprocess = tio.Compose([
            tio.CropOrPad((h, w, d), padding_mode=0),
            tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # Use the named function instead of a lambda
            tio.Lambda(apply_hist_norm, types_to_apply=[tio.INTENSITY])  # Apply only to intensity images
        ])
    else:
        print('dummy')


def get_eval_transform(cfg): # only transforms that are applied once before preloading
    exclude_from_resampling = None
    h, w, d = tuple(cfg.get('imgDimResize',(160,192,160)))
    if cfg.get('unisotropic_sampling',True):
        preprocess = tio.Compose([
        tio.CropOrPad((h,w,d),padding_mode=0),
        #tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling)
        ,#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        print('dummy')

    return preprocess

def get_eval_transform_v2(cfg):
    exclude_from_resampling = None
    h, w, d = tuple(cfg.get('imgDimResize', (160, 192, 160)))
    
    # Initialize the MONAI HistogramNormalize transform
    hist_norm = monai.transforms.HistogramNormalize(num_bins=256,min=0,max=1)
    
    if cfg.get('unisotropic_sampling', True):
        preprocess = tio.Compose([
            tio.CropOrPad((h, w, d), padding_mode=0),
            # tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)), masking_method='mask'),
            tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # Add MONAI HistogramNormalize here
            tio.Lambda(lambda x: hist_norm(x), types_to_apply=[tio.INTENSITY])  # Apply only to intensity images
        ])
    else:
        print('dummy')

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
        subject['vol'].data[subject['vol'].data<0.01]=0
        subject['vol'].data = subject['vol'].data.permute(0,3, 2, 1)
        msk_normal = (np.count_nonzero(subject['vol'].data.numpy()[0],axis=(1,2))/(subject['vol'].data.numpy()[0].shape[1]*subject['vol'].data.numpy()[0].shape[2]))>=0.15
        

        choices = np.arange(len(msk_normal))[msk_normal]
        sample_idx = np.array(random.choices(choices,k = 1))
        subject['vol'].data = subject['vol'].data[0:2, sample_idx, ...]
        return subject

def get_augment(cfg): # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment

