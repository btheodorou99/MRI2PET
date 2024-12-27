import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.autograd as autograd

class Linear_kml(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_kml, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.u_vector = nn.Parameter(torch.randn(in_features) * 1e-7)
        self.v_vector = nn.Parameter(torch.randn(out_features) * 1e-7)
        if bias:
            self.b_vector = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter('b_vector', None)
        
    def forward(self, input):
        W_hat  = torch.ger(self.u_vector, self.v_vector).view(self.W.shape)
        weight = self.W * (torch.ones_like(self.W) + W_hat)
        bias   = self.bias + self.b_vector
        out = F.linear(input, weight, bias=bias)
        return out

class Conv3d_kml(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv3d_kml, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as tuple or single number
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        kernel_size_prod = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.u_vector = nn.Parameter(torch.randn(in_channels * kernel_size_prod) * 1e-7)
        self.v_vector = nn.Parameter(torch.randn(out_channels) * 1e-7)
        if bias:
            self.b_vector = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('b_vector', None)

    def forward(self, input):
        W_hat  = torch.ger(self.u_vector, self.v_vector).view(self.W.shape)
        weight = self.W * (torch.ones_like(self.W) + W_hat)
        bias   = self.bias + self.b_vector
        out = F.conv3d(input, weight, bias=bias, stride=self.stride, padding=self.padding)
        return out

class ConvTranspose3d_kml(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvTranspose3d_kml, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.W = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        kernel_size_prod = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.u_vector = nn.Parameter(torch.randn(in_channels * kernel_size_prod) * 1e-7)
        self.v_vector = nn.Parameter(torch.randn(out_channels) * 1e-7)
        if bias:
            self.b_vector = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('b_vector', None)

    def forward(self, input):
        W_hat  = torch.ger(self.u_vector, self.v_vector).view(self.W.shape)
        weight = self.W * (torch.ones_like(self.W) + W_hat)
        bias   = self.bias + self.b_vector
        out = F.conv_transpose3d(input, weight, bias=bias, stride=self.stride, padding=self.padding)
        return out

class ImageEncoder(nn.Module):
    def __init__(self, config, is_mri=True):
        super(ImageEncoder, self).__init__()
        self.is_mri = is_mri
        self.depth = config.n_mri_channels if self.is_mri else config.n_pet_channels
        self.image_dim = config.mri_image_dim if self.is_mri else config.pet_image_dim

        self.conv1 = Conv3d_kml(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv3d_kml(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv3d_kml(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv3d_kml(32, 64, kernel_size=3, stride=1, padding=1)
        self.flat_dim = 64 * (self.depth // 16) * (self.image_dim // 16) * (self.image_dim // 16)
        self.fc1 = Linear_kml(self.flat_dim, config.embed_dim)
        self.fc2 = Linear_kml(config.embed_dim, config.embed_dim)

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
        self.init_depth = config.n_pet_channels // (2 * 2 * 3 * 1 * 1)
        self.z_dim = config.z_dim

        self.context_emb = ImageEncoder(config, is_mri=True)
        self.init_map = nn.Linear(config.z_dim + config.embed_dim, 8 * self.init_depth * (self.init_dim ** 2))

        self.gen = nn.Sequential(
            self._gen_block(8, 16),
            self._gen_block(16, 32),
            self._gen_block(32, 32),
            self._gen_block(32, 16),
            self._gen_block_single(16, 8),
            Conv3d_kml(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv3d_kml(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def _gen_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            ConvTranspose3d_kml(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _gen_block_single(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            ConvTranspose3d_kml(in_channels, out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, noise, context_images):
        context_images = context_images.unsqueeze(1)
        context = self.context_emb(context_images)
        gen_input = torch.cat((context, noise), -1)
        gen_input = self.init_map(gen_input)
        gen_input = gen_input.view(gen_input.size(0), 8, self.init_depth, self.init_dim, self.init_dim)
        img = self.gen(gen_input)
        img = img.squeeze(1)
        return img

    def estimate_fisher(self, loglikelihood):
        """
        Implementation from GAN Memory: https://arxiv.org/pdf/2006.07543.pdf, NeurIPS 2020.

        input: a loglikelihood (generator loss) from a *batch* of generated images (output of the discriminator)
        output: estimated fisher information for each trainable parameter in generator
        """
        # Initialize fisher information placeholder
        est_fisher_info = {} # dict
        for n, p in self.named_parameters():
            if p.requires_grad:
                est_fisher_info[n] = p.detach().clone().zero_()

        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters(), retain_graph=True) 

        # Square gradients and return
        for i, (n, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                if loglikelihood_grads[i] is not None:
                    est_fisher_info[n] = loglikelihood_grads[i].detach() ** 2

        return loglikelihood_grads, est_fisher_info

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
            Linear_kml(2*config.embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            Linear_kml(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            Linear_kml(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            Linear_kml(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            Linear_kml(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            Linear_kml(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, context_images):
        img = img.unsqueeze(1)
        context_images = context_images.unsqueeze(1)
        context = self.context_emb(context_images)
        image = self.image_emb(img)
        disc_input = torch.cat((context, image), -1)
        validity = self.model(disc_input)
        return validity

    def estimate_fisher(self, loglikelihood):
        """
        Implementation from GAN Memory: https://arxiv.org/pdf/2006.07543.pdf, NeurIPS 2020.

        input: a loglikelihood from a *batch* of generated images (output of the discriminator)
        output: estimated fisher information for each trainable parameter in generator
        """
        # Initialize fisher information placeholder
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                est_fisher_info[n] = p.detach().clone().zero_()

        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters()) 

        # Square gradients and return
        for i, (n, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                if loglikelihood_grads[i] is not None:
                    est_fisher_info[n] = loglikelihood_grads[i].detach() ** 2

        return loglikelihood_grads, est_fisher_info

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
