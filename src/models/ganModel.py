import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, config, is_mri=True):
        super(ImageEncoder, self).__init__()
        self.n_channels = config.n_mri_channels if is_mri else config.n_pet_channels
        self.image_dim = config.mri_image_dim if is_mri else config.pet_image_dim

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

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.init_dim = config.pet_image_dim // 32
        self.z_dim = config.z_dim

        self.context_emb = ImageEncoder(config, is_mri=True)
        self.init_map = nn.Linear(config.z_dim + config.embed_dim, 32 * (self.init_dim ** 2))

        self.gen = nn.Sequential(
            self._gen_block(32, 64),
            self._gen_block(64, 128),
            self._gen_block(128, 128),
            self._gen_block(128, 64),
            self._gen_block(64, 32),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, config.n_pet_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def _gen_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, noise, context_images):
        context = self.context_emb(context_images)
        gen_input = torch.cat((context, noise), -1)
        gen_input = self.init_map(gen_input)
        gen_input = gen_input.view(gen_input.size(0), 32, self.init_dim, self.init_dim)
        img = self.gen(gen_input)
        return img
    
    def generate(self, context_images):
        z = torch.randn(context_images.size(0), self.z_dim, device=context_images.device)
        images = self.forward(z, context_images)
        return images

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.context_emb = ImageEncoder(config, is_mri=True)
        self.image_emb = ImageEncoder(config, is_mri=False)
        self.model = nn.Sequential(
            nn.Linear(2*config.embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, context_images):
        context = self.context_emb(context_images)
        image = self.image_emb(img)
        disc_input = torch.cat((context, image), -1)
        validity = self.model(disc_input)
        return validity

    def compute_gradient_penalty(self, real_samples, fake_samples, context_images):
        # Interpolate real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.forward(interpolates, context_images)
        fake = torch.ones(d_interpolates.shape, requires_grad=False, device=real_samples.device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty