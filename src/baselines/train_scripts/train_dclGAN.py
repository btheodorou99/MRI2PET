# A Closer Look at Few-shot Image Generation

import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
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

pretrain_dataset = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
pretrain_dataset = [(mri_path, os.path.join(config.mri_pretrain_dir, mri_path.split('/')[-1])) for mri_path in pretrain_dataset]
train_dataset = pickle.load(open('./src/data/trainDataset.pkl', 'rb'))
val_dataset = pickle.load(open('./src/data/valDataset.pkl', 'rb'))
DCL_TAU = 0.07
DCL_LAMBDA1 = 2
DCL_LAMBDA2 = 0.5

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if is_mri:
        img += torch.randn(img.size())
    else:
        img = 2 * img - 1
    return img

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
if os.path.exists(f"./src/save/dclGAN.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/dclGAN.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_g'])
    optimizer_D.load_state_dict(checkpoint['optimizer_d'])

steps_per_batch = 4
config.batch_size = config.batch_size // steps_per_batch

for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(pretrain_dataset)
    generator.train()
    discriminator.train()
    curr_step = 0
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()
    for i in range(0, len(pretrain_dataset), config.batch_size):
        batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)

        # Train Discriminator
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        fake_imgs = generator(z, batch_context)

        real_validity = discriminator(batch_images, batch_context)
        fake_validity = discriminator(fake_imgs, batch_context)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty
        d_loss = d_loss / steps_per_batch
        d_loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizer_D.step()
            optimizer_D.zero_grad()

        if i % (config.generator_interval * config.batch_size) == 0:
            # Train Generator
            fake_imgs = generator(z, batch_context)
            fake_validity = discriminator(fake_imgs, batch_context)
            g_loss = -torch.mean(fake_validity)
            g_loss = g_loss / steps_per_batch
            g_loss.backward()
            if curr_step % (steps_per_batch * config.generator_interval) == 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
                optimizer_G.step()
                optimizer_G.zero_grad()
                curr_step = 0

    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'./src/save/dclGAN_base.pt')

G_s = deepcopy(generator).eval().requires_grad_(False)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)

for e in tqdm(range(config.epoch*config.generator_interval)):
    shuffle_training_data(train_dataset)
    generator.train()
    discriminator.train()
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()
    curr_step = 0
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        if len(batch_context) == 1:
            continue
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        
        # Train Discriminator
        fake_imgs = generator(z, batch_context)
        acts_R, real_validity = discriminator(batch_images, batch_context, finetune=True)
        acts_T, fake_validity = discriminator(fake_imgs, batch_context, finetune=True)
        acts_S, _ = discriminator(G_s(z, batch_context), batch_context, finetune=True)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        dcl_loss = discriminator.compute_dcl_loss(acts_S, acts_T, acts_R, DCL_TAU)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty + DCL_LAMBDA2 * dcl_loss
        d_loss = d_loss / steps_per_batch
        d_loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizer_D.step()
            optimizer_D.zero_grad()

        if i % (config.generator_interval * config.batch_size) == 0:
            print("Train Generator", curr_step, steps_per_batch*config.generator_interval)
            # Train Generator
            acts_T, fake_imgs = generator(z, batch_context, finetune=True)
            acts_S, _ = G_s(z, batch_context, finetune=True)
            fake_validity = discriminator(fake_imgs, batch_context, finetune=False)
            g_loss = -torch.mean(fake_validity)
            g_loss += DCL_LAMBDA1 * generator.compute_dcl_loss(acts_S, acts_T, DCL_TAU)
            g_loss = g_loss / steps_per_batch
            g_loss.backward()
            if curr_step % (steps_per_batch * config.generator_interval) == 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
                optimizer_G.step()
                optimizer_G.zero_grad()
                curr_step = 0

    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'./src/save/dclGAN.pt')
