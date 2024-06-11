import math
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, config, final=False, noise=False, timestep=0):
        super(DiffusionModel, self).__init__()
        self.num_timesteps = config.num_timesteps
        self.beta = torch.linspace(config.beta_start, config.beta_end, self.num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.image_dim = config.pet_image_dim
        self.n_channels = config.n_pet_channels
        self.embed_dim = 32
        self.final = final
        self.noise = noise
        self.timestep = timestep
        self.input_dim = config.n_pet_channels if self.timestep == 0 else config.n_pet_channels + self.embed_dim
        if self.timestep == 3:
            self.time_emb = nn.Sequential(nn.Linear(self.embed_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.embed_dim))

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.n_pet_channels, kernel_size=3, padding=1)
        )

    def timestep_embedding1(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(self.num_timesteps) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb[:, :, None, None].expand(-1, -1, self.image_dim, self.image_dim)
        return emb
    
    def timestep_embedding2(self, t):
        inv_freq = 1.0 / (
            self.num_timesteps 
                ** (torch.arange(0, self.embed_dim, 2, device=t.device).float() / self.embed_dim)
        )
        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, self.embed_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, self.embed_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        pos_enc = pos_enc[:, :, None, None].expand(-1, -1, self.image_dim, self.image_dim)
        return pos_enc

    def forward(self, context, x, gen_loss=True):
        t = torch.randint(1, self.num_timesteps, (x.size(0),), device=x.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(x.device)
        noise = torch.randn_like(x)
        if self.noise:
            noise.clamp_(-1, 1)
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        if self.timestep == 1:
            x_noisy = torch.cat((x_noisy, self.timestep_embedding1(t)), dim=1)
        elif self.timestep == 2:
            x_noisy = torch.cat((x_noisy, self.timestep_embedding2(t)), dim=1)
        elif self.timestep == 3:
            x_noisy = torch.cat((x_noisy, self.time_emb(self.timestep_embedding1(t))), dim=1)
        x_pred = self.encoder(x_noisy)
        if self.final:
            x_pred = 2 * torch.tanh(x_pred)
        if gen_loss:
            return self.loss(x, x_pred, noise), x_pred
        return x_pred

    def loss(self, x, x_pred, noise):
        return nn.MSELoss()(x_pred, noise)

    def generate(self, context):
        n = context.size(0)
        x = torch.randn(n, self.n_channels, self.image_dim, self.image_dim, device=context.device)
        for timestep in reversed(range(1, self.num_timesteps)):
            t = torch.ones(n).long() * timestep
            x_pred = self.encoder(x)
            alpha = self.alpha[t][:, None, None, None].to(context.device)
            alpha_hat = self.alpha_hat[t][:, None, None, None].to(context.device)
            beta = self.beta[t][:, None, None, None].to(context.device)
            if timestep > 1:
                noise = torch.randn_like(x, device=context.device)
            else:
                noise = torch.zeros_like(x, device=context.device)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * x_pred) + torch.sqrt(beta) * noise
        return x.clamp(-1, 1)
