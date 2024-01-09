# A Closer Look at Few-shot Image Generation

import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from ...config import MRI2PETConfig
from ..models.dclGAN import Generator, Discriminator

SEED = 4
cudaNum = 0
NUM_SAMPLES = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

pretrain_dataset = pickle.load(open('../../data/pretrainDataset.pkl', 'rb'))
pretrain_dataset = [(mri_path, mri_path) for (mri_path, style_transfer_path) in pretrain_dataset]
train_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))
train_dataset = [(mri_path, pet_path) for (mri_path, pet_path) in train_dataset]
val_dataset = pickle.load(open('../../data/valDataset.pkl', 'rb'))
val_dataset = [(mri_path, pet_path) for (mri_path, pet_path) in val_dataset]
DCL_TAU = 0.07
DCL_LAMBDA1 = 2
DCL_LAMBDA2 = 0.5

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


image_transform_mri = transforms.Compose([
        transforms.Resize((config.mri_image_dim, config.mri_image_dim)),
        transforms.ToTensor(),
        AddGaussianNoise(0., 1.),
        # transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])
image_transform_pet = transforms.Compose([
        transforms.Resize((config.pet_image_dim, config.pet_image_dim)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])

def load_image(image_path, is_mri=True):
    with Image.open(f'{config.image_dir}/{image_path}.jpg') as img:
        return image_transform_mri(img) if is_mri else image_transform_pet(img)

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float, device=device)
    batch_image = torch.zeros(bs, config.n_pet_channels, config.pet_image_dim, config.pet_image_dim, dtype=torch.float, device=device)
    for i, (m, p) in enumerate(image_paths):
        batch_context[i] = load_image(m, is_mri=True)
        batch_image[i] = load_image(p, is_mri=False)
        
    return batch_context, batch_image

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
if os.path.exists(f"../../save/dclGAN.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'../../save/dclGAN.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_g'])
    optimizer_D.load_state_dict(checkpoint['optimizer_d'])

for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(pretrain_dataset)
    generator.train()
    discriminator.train()
    for i in range(0, len(pretrain_dataset), config.batch_size):
        batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)
        
        # Train Discriminator
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        fake_imgs = generator(z, batch_context)

        real_validity = discriminator(batch_images, batch_context)
        fake_validity = discriminator(fake_imgs, batch_context)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        if i % (config.generator_interval * config.batch_size) == 0:
            # Train Generator
            fake_imgs = generator(z, batch_context)
            fake_validity = discriminator(fake_imgs, batch_context)
            g_loss = -torch.mean(fake_validity)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'../../save/dclGAN.pt')

G_s = deepcopy(generator).eval().requires_grad_(False)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    generator.train()
    discriminator.train()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        
        # Train Discriminator
        fake_imgs = generator(z, batch_context)
        acts_R, real_validity = discriminator(batch_images, batch_context, finetune=True)
        acts_T, fake_validity = discriminator(fake_imgs, batch_context, finetune=True)
        acts_S, _ = discriminator(G_s(z, batch_context), batch_context, finetune=True)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        dcl_loss = discriminator.compute_dcl_loss(acts_S, acts_T, acts_R, DCL_TAU)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty + DCL_LAMBDA2 * dcl_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        if i % (config.generator_interval * config.batch_size) == 0:
            # Train Generator
            acts_T, fake_imgs = generator(z, batch_context, finetune=True)
            acts_S, _ = G_s(z, batch_context, finetune=True)
            fake_validity = discriminator(fake_imgs, batch_context, finetune=True)
            g_loss = -torch.mean(fake_validity)
            g_loss += DCL_LAMBDA1 * generator.compute_dcl_loss(acts_S, acts_T, DCL_TAU)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'../../save/dclGAN.pt')