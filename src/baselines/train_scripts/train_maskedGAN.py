# Few-shot Image Generation via Masked Discrimination

import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ...config import MRI2PETConfig
from ..models.maskedGAN import Generator, Discriminator

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
CDC_LAMBDA = 1e3
SUBSPACE_FREQ = 4
SUBSPACE_STD = 0.1

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

def get_subspace(init_z, bs):
    std = SUBSPACE_STD
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z.to(device)

generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
if os.path.exists(f"./src/save/maskedGAN.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/maskedGAN.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    start_epoch = checkpoint['epoch']

    G_s = Generator(config).to(device)
    D_s = Discriminator(config).to(device)
    checkpoint_s = torch.load(f'./src/save/cdcGAN_base.pt', map_location=torch.device(device))
    G_s.load_state_dict(checkpoint_s['generator'])
    D_s.load_state_dict(checkpoint_s['discriminator'])
else:
    start_epoch = -1

steps_per_batch = 4
config.batch_size = config.batch_size // steps_per_batch

# for e in tqdm(range(config.pretrain_epoch)):
#     shuffle_training_data(pretrain_dataset)
#     generator.train()
#     discriminator.train()
#     curr_step = 0
#     optimizer_D.zero_grad()
#     optimizer_G.zero_grad()
#     for i in range(0, len(pretrain_dataset), config.batch_size):
#         batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)

#         # Train Discriminator
#         z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
#         fake_imgs = generator(z, batch_context)

#         real_validity = discriminator(batch_images, batch_context)
#         fake_validity = discriminator(fake_imgs, batch_context)
#         gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
#         d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty
#         d_loss = d_loss / steps_per_batch
#         d_loss.backward()
#         curr_step += 1
#         if curr_step % steps_per_batch == 0:
#             torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
#             optimizer_D.step()
#             optimizer_D.zero_grad()

#         if i % (config.generator_interval * config.batch_size) == 0:
#             # Train Generator
#             fake_imgs = generator(z, batch_context)
#             fake_validity = discriminator(fake_imgs, batch_context)
#             g_loss = -torch.mean(fake_validity)
#             g_loss = g_loss / steps_per_batch
#             g_loss.backward()
#             if (curr_step + 1) % (steps_per_batch * config.generator_interval) == 0:
#                 torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
#                 optimizer_G.step()
#                 optimizer_G.zero_grad()
#                 curr_step = 0

#     state = {
#         'generator': generator.state_dict(),
#         'discriminator': discriminator.state_dict(),
#         'optimizer_G': optimizer_G,
#         'optimizer_D': optimizer_D,
#         'epoch': e
#     }
#     torch.save(state, f'./src/save/maskedGAN_base.pt')

# G_s = deepcopy(generator).eval().requires_grad_(False)
# D_s = deepcopy(discriminator).eval().requires_grad_(False)
init_z = torch.randn(len(train_dataset), config.z_dim, device=device)
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)

for e in tqdm(range(start_epoch+1, config.epoch)):
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
        
        which = (i // config.batch_size) % SUBSPACE_FREQ
        if which == 0:
            z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        else:
            z = get_subspace(init_z, batch_context.size(0))
        
        # Train Discriminator
        fake_imgs = generator(z, batch_context)
        _, real_validity = discriminator(batch_images, batch_context, finetune=True, flag=which)
        acts_T, fake_validity = discriminator(fake_imgs, batch_context, finetune=True, flag=which)
        acts_S, _ = D_s(G_s(z, batch_context), batch_context, finetune=True, flag=which)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        cdc_loss = discriminator.compute_cdc_loss(acts_S, acts_T)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty + CDC_LAMBDA * cdc_loss
        d_loss = d_loss / steps_per_batch
        d_loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizer_D.step()
            optimizer_D.zero_grad()

        if i % (config.generator_interval * config.batch_size) == 0:
            acts_T, fake_imgs = generator(z, batch_context, finetune=True)
            acts_S, _ = G_s(z, batch_context, finetune=True)
            fake_validity = discriminator(fake_imgs, batch_context, finetune=False)
            g_loss = -torch.mean(fake_validity)
            g_loss += CDC_LAMBDA * generator.compute_cdc_loss(acts_S, acts_T)
            g_loss = g_loss / steps_per_batch
            g_loss.backward()
            if (curr_step + 1) % (steps_per_batch * config.generator_interval) == 0:
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
    torch.save(state, f'./src/save/maskedGAN.pt')
