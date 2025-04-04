import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig
from ..models.ganModel import Generator, Discriminator

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

train_dataset = pickle.load(open('./src/data/trainDataset.pkl', 'rb'))
val_dataset = pickle.load(open('./src/data/valDataset.pkl', 'rb'))

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if not is_mri:
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
if os.path.exists(f"./src/save/baseGAN.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/baseGAN.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    generator.train()
    discriminator.train()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        
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
    torch.save(state, f'./src/save/baseGAN.pt')