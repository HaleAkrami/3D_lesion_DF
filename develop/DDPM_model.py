import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial

from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from PIL import Image

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


#from src.utils.generate_noise import gen_noise
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        inpaint = False,
        cfg=None,
    ):
        super().__init__()
        if not hasattr(model,'channels'):
            pass
        else: 
            assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        self.cfg = cfg
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.objective = objective
        self.inpaint = inpaint
        self.use_spatial_transformer = cfg.use_spatial_transformer
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond, cond_scale, clip_x_start=False):
        if self.use_spatial_transformer:
            if x.shape[1] == 2: # if channel conditioning is used, we indicate the patch by ones
                x[:,1][x[:,1]!=-1]=1
            model_output = self.model.forward_with_cond_scale(x, t, cond = cond, context = cond.unsqueeze(1), cond_scale = cond_scale) # predict the noise that has been added to x_start or directly predict x_start from the noisy x, conditioned by the timestep t, cond
            if x.shape[1] == 2: 
                x = x[:,0].unsqueeze(1)
        else: 
            model_output = self.model.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale) # predict the noise that has been added to x_start or directly predict x_start from the noisy x, conditioned by the timestep t, cond
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
            x_start = maybe_clip(x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        preds = self.model_predictions(x, t, cond, cond_scale, clip_denoised)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t: int, clip_denoised = True, cond = None, cond_scale = 1., noise = None):
        b, *_, device = *x.shape, x.device

        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = batched_times, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        if noise is None:
            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        else:
            dummy =0
            # noise = gen_noise(self.cfg, x.shape).to(device)
            # noise = noise.float() if t > 0 else 0.
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1., box = None, start_t = 0, noise = None, x_start=None):
        batch, device = shape[0], self.betas.device
        T = self.num_timesteps if start_t==0 else start_t
        if noise is not None: 
            dummy = 0
            # noise = gen_noise(self.cfg, shape).to(device)
            # img = self.q_sample(x_start = x_start, t = torch.tensor([T],device=device), noise = noise)[:,0].unsqueeze(1)
        else:
            img = torch.randn(shape, device=device) # generate noisy image x_T
        if box is not None:
            img_patch = torch.zeros_like(img)
            for i in range(img.shape[0]):
                img_patch[i, :, box[i,1]:box[i,3], box[i,0]:box[i,2]] = img[i, :, box[i,1]:box[i,3], box[i,0]:box[i,2]]
                img = img_patch
        for t in reversed(range(0, T)): # T -> 0
            img = self.p_sample(img, t, cond = cond, cond_scale = cond_scale, noise = noise)

        img = unnormalize_to_zero_to_one(img)
        return img

  

    @torch.no_grad()
    def sample(self, batch_size = 1, cond = None, cond_scale = 1., box = None, x_start = None, start_t = 0, noise=None):

        batch_size = x_start.shape[0] if exists(cond) else batch_size
        
        image_size_h,image_size_w, channels = self.image_size[0], self.image_size[1], self.channels
        
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        if box is not None: 
            sample_fn = self.ddim_sample_box
            return sample_fn((batch_size, channels, image_size_h, image_size_w), x_start, cond = cond, cond_scale = cond_scale, box = box, start_t = start_t, noise=noise)
        else : 
            return sample_fn((batch_size, channels, image_size_h, image_size_w), cond = cond, cond_scale = cond_scale, start_t = start_t, noise=noise,x_start=x_start)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, cond = None, noise = None, box=None, scale_patch=1, onlybox=False, mask=None):
        b, c, h, w,z = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start)) # some noise with zero mean and unit variance

        x = self.q_sample(x_start = x_start, t = t, noise = noise) # generate a noisy image from the start image at timestep t
        
        


        model_out = self.model(x, t, cond = None) # predict the noise that has been added to x_start or directly predict x_start from the noisy x, conditioned by the timestep t, cond

        if self.objective == 'pred_noise': # predict the noise that has been added to x_start
            target = noise

        elif self.objective == 'pred_x0': # directly predict x_start from the noisy image x
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')


        loss = self.loss_fn(model_out, target, reduction = 'none')   
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        if self.objective == 'pred_noise':
            return loss.mean(), unnormalize_to_zero_to_one(x - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * model_out) 
        else:
            return loss.mean(), unnormalize_to_zero_to_one(model_out)

    def forward(self, img, t=None, *args, **kwargs):
        b, c, h,w,z, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() if t is None else (torch.ones([b],device=device)*t).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


   