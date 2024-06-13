# Wavelet Scattering Transform Loss to Emphasize Details

import torch 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat
from kymatio.torch import Scattering2D

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.n_channels = config.n_mri_channels
        self.image_dim = config.mri_image_dim

        self.conv1 = nn.Conv2d(self.n_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.flat_dim = 256 * (self.image_dim // 16) * (self.image_dim // 16)
        self.fc1 = nn.Linear(self.flat_dim, config.embed_dim)
        self.fc2 = nn.Linear(config.embed_dim, config.embed_dim)
        
    def forward(self, x):        
        # Convolution + ReLU + MaxPooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        # Flattening the output
        x = x.view(-1, self.flat_dim)

        # Passing through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
    
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

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
        self.lambda1 = 0.25
        
        self.contextEmbedding = ImageEncoder(config)
        self.inc = DoubleConv(config.n_pet_channels, 16)
        
        self.down1 = DownBlock(16, 32, self.embed_dim)
        self.sa1 = LinearAttention(32)
        self.down2 = DownBlock(32, 64, self.embed_dim)
        self.sa2 = LinearAttention(64)
        self.down3 = DownBlock(64, 128, self.embed_dim)
        self.sa3 = LinearAttention(128)
        self.down4 = DownBlock(128, 256, self.embed_dim)
        self.sa4 = LinearAttention(256)
        
        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 256)
        
        self.up1 = UpBlock(384, 128, self.embed_dim)
        self.sa5 = LinearAttention(128)
        self.up2 = UpBlock(192, 64, self.embed_dim)
        self.sa6 = LinearAttention(64)
        self.up3 = UpBlock(96, 32, self.embed_dim)
        self.sa7 = LinearAttention(32)
        self.up4 = UpBlock(48, 16, self.embed_dim)
        self.sa8 = LinearAttention(16)
        self.outc = nn.Conv2d(16, config.n_pet_channels, kernel_size=1)

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
        emb = self.timestep_embedding(t, self.embed_dim, max_period=self.num_timesteps)
        emb += self.contextEmbedding(condImage)
        
        x1 = self.inc(noised_images)
        x2 = self.down1(x1, emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, emb)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, emb)
        x5 = self.sa4(x5)
        
        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)
        
        x = self.up1(x5, x4, emb)
        x = self.sa5(x)
        x = self.up2(x, x3, emb)
        x = self.sa6(x)
        x = self.up3(x, x2, emb)
        x = self.sa7(x)
        x = self.up4(x, x1, emb)
        x = self.sa8(x)
        x = self.outc(x)
        return x
    
    def construct_image(self, noised_x, t, pred_noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(noised_x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(noised_x.device)
        pred_x = (1 / sqrt_alpha_hat) * noised_x - (sqrt_one_minus_alpha_hat / sqrt_alpha_hat) * pred_noise
        return pred_x

    def wavelet_scattering_transform_loss(img1, img2, J=2):
        # Initialize scattering transform
        scattering = Scattering2D(J=J, shape=img1.shape[-2:])
        scattering = scattering.to(img1.device)

        # Apply scattering transform
        S_img1 = scattering(img1)
        S_img2 = scattering(img2)

        # Calculate loss
        loss = F.mse_loss(S_img1, S_img2)

        return loss
    
    def forward(self, context, input_images, gen_loss=True):
        t = self.sample_timesteps(input_images.size(0), context.device)
        noised_images, noise = self.noise_images(input_images, t)
        predictedNoise = self._forward(noised_images, t, context)
        if gen_loss:
            predictedImage = self.construct_image(noised_images, t, predictedNoise)
            loss = F.mse_loss(noise, predictedNoise)
            wavelet_loss = self.wavelet_scattering_transform_loss(input_images, predictedImage)
            return loss + self.lambda1 * wavelet_loss, predictedNoise
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