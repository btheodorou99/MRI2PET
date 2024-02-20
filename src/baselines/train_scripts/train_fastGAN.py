import os
import lpips
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from ...config import MRI2PETConfig
from ..models.diffAugmentGAN import DiffAugment
from ..models.fastGAN import weights_init, Generator, Discriminator

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
train_dataset = [(os.path.join(config.mri_image_dir, mri_path), os.path.join(config.pet_image_dir, pet_path)) for (mri_path, pet_path) in train_dataset]
val_dataset = pickle.load(open('./src/data/valDataset.pkl', 'rb'))
val_dataset = [(os.path.join(config.mri_image_dir, mri_path), os.path.join(config.pet_image_dir, pet_path)) for (mri_path, pet_path) in val_dataset]

policy = 'color,translation'
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

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
    
def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
  
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)
   
def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]
        
def train_d(net, data, context, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, context, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, context, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

generator = Generator(config).to(device)
generator.apply(weights_init)
discriminator = Discriminator(config).to(device)
discriminator.apply(weights_init)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
if os.path.exists(f"./src/save/fastGAN.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/fastGAN.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_g'])
    optimizer_D.load_state_dict(checkpoint['optimizer_d'])
    
avg_param_G = copy_G_params(generator)
fixed_noise = torch.FloatTensor(8, config.z_dim).normal_(0, 1).to(device)

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    generator.train()
    discriminator.train()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        fake_imgs = generator(z, batch_context)
        
        batch_images = DiffAugment(batch_images, policy=policy)
        fake_imgs = [DiffAugment(fake, policy=policy) for fake in fake_imgs]
        
        # Train Discriminator
        discriminator.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(discriminator, batch_images, batch_context, label="real")
        train_d(discriminator, [fi.detach() for fi in fake_imgs], batch_context, label="fake")
        optimizer_D.step()
        
        # Train Generator
        generator.zero_grad()
        pred_g = discriminator(fake_imgs, batch_context, label="fake")
        err_g = -pred_g.mean()
        err_g.backward()
        optimizer_G.step()
        
        for p, avg_p in zip(generator.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

    backup_para = copy_G_params(generator)
    load_params(generator, avg_param_G)
    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'./src/save/fastGAN.pt')
    load_params(generator, backup_para)