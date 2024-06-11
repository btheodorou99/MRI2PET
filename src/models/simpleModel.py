import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.n_channels = config.n_mri_channels
        self.image_dim = config.mri_image_dim

        self.conv1 = nn.Conv2d(self.n_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flat_dim = 256 * (self.image_dim // 4) * (self.image_dim // 4)
        self.fc1 = nn.Linear(self.flat_dim, config.embed_dim)
        self.fc2 = nn.Linear(config.embed_dim, config.mri_image_dim * config.mri_image_dim)
        
    def forward(self, x):        
        # Convolution + ReLU + MaxPooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flattening the output
        x = x.view(-1, self.flat_dim)

        # Passing through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = config.num_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.image_dim = config.pet_image_dim
        self.n_channels = config.n_pet_channels
        self.embed_dim = config.embed_dim
        
        self.contextEmbedding = ImageEncoder(config)
        self.conv1 = nn.Conv2d(self.n_channels + 1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, self.n_channels, kernel_size=3, stride=1, padding=1)


    def timestep_embedding(self, t, channels, max_period=100):
        inv_freq = 1.0 / (
            max_period 
             ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def sample_timesteps(self, n, device='cpu'):
        return torch.randint(low=1, high=self.num_timesteps, size=(n,), device=device)

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(x.device)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def _forward(self, noised_images, t, condImage):
        "Forward pass through the model"
        emb = self.timestep_embedding(t, self.image_dim * self.image_dim, max_period=self.num_timesteps)
        emb += self.contextEmbedding(condImage)
        x = torch.cat([noised_images, emb.view(-1, 1, self.image_dim, self.image_dim)], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
    
    def forward(self, context, input_images, gen_loss=True):
        t = self.sample_timesteps(input_images.size(0), context.device)
        noised_images, noise = self.noise_images(input_images, t)
        predictedNoise = self._forward(noised_images, t, context)
        if gen_loss:
            loss = F.mse_loss(noise, predictedNoise)
            return loss, predictedNoise
        return predictedNoise

    def generate(self, context):
        n = context.size(0)
        x = torch.randn(n, self.n_channels, self.image_dim, self.image_dim, device=context.device)
        for timestep in tqdm(reversed(range(1, self.num_timesteps)), total=self.num_timesteps-1):
            t = (torch.ones(n) * timestep).long().to(context.device)
            predicted_noise = self._forward(x, t, context)
            alpha = self.alpha[t][:, None, None, None].to(context.device)
            alpha_hat = self.alpha_hat[t][:, None, None, None].to(context.device)
            beta = self.beta[t][:, None, None, None].to(context.device)
            if timestep > 1:
                noise = torch.randn_like(x, device=context.device)
            else:
                noise = torch.zeros_like(x, device=context.device)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = x.clamp(-1,1)
        return x
    
    def calculate_nll(self, context, input_images):
        nll = 0.0
        for timestep in range(1, self.num_timesteps):
            t = torch.ones(input_images.size(0), device=context.device) * timestep
            noised_images, actual_noise = self.noise_images(input_images, t)
            predicted_noise = self._forward(noised_images, t, context)
            
            # Calculate the mean squared error between predicted and actual noise
            mse = torch.mean((predicted_noise - actual_noise) ** 2)

            # Convert MSE to log-likelihood assuming Gaussian noise
            sigma_t = torch.sqrt(1 - self.alpha_hat[t])
            log_likelihood = -mse / (2 * sigma_t**2) - 0.5 * np.log(2 * np.pi * sigma_t**2)

            # Accumulate NLL
            nll += log_likelihood
        
        return nll.item() * input_images.numel()