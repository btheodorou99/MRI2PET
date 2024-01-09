import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from ..config import MRI2PETConfig
from ..models.proposedModel7 import DiffusionModel

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

pretrain_dataset = pickle.load(open('../data/pretrainDataset.pkl', 'rb'))
pretrain_dataset = [(mri_path, style_transfer_path) for (mri_path, style_transfer_path) in pretrain_dataset]
train_dataset = pickle.load(open('../data/trainDataset.pkl', 'rb'))
train_dataset = [(mri_path, pet_path) for (mri_path, pet_path) in train_dataset]
val_dataset = pickle.load(open('../data/valDataset.pkl', 'rb'))
val_dataset = [(mri_path, pet_path) for (mri_path, pet_path) in val_dataset]
image_transform_mri = transforms.Compose([
        transforms.Resize((config.mri_image_dim, config.mri_image_dim)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])
image_transform_pet = transforms.Compose([
        transforms.Resize((config.pet_image_dim, config.pet_image_dim)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])
MRI_LOSS_SCHEDULE = [max(0.5 - (0.5 * e / (config.epoch // 4)), 0) for e in config.epoch]

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

model = DiffusionModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(f"../save/proposedModel7.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'../save/proposedModel7.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(pretrain_dataset)
    pretrain_losses = []
    model.train()
    for i in range(0, len(pretrain_dataset), config.batch_size):
        batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)
        optimizer.zero_grad()
        loss, _ = model(batch_context, batch_images, gen_loss=True)
        loss.backward()
        optimizer.step()
        pretrain_losses.append(loss.cpu().detach().item())
    
    cur_pretrain_loss = np.mean(pretrain_losses)
    print("Epoch %d Training Loss:%.7f"%(e, cur_pretrain_loss))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mode': 'pretrain'
    }
    torch.save(state, f'../save/proposedModel7.pt')

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    train_losses = []
    model.train()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        optimizer.zero_grad()
        loss, _ = model(batch_context, batch_images, gen_loss=True)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.cpu().detach().item())
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for v_i in range(0, len(val_dataset), config.batch_size):
            batch_context, batch_images = get_batch(val_dataset, v_i, config.batch_size)                 
            val_loss, _ = model(batch_context, batch_images, gen_loss=True, weight=MRI_LOSS_SCHEDULE[e])
            val_losses.append((val_loss).cpu().detach().item())
        
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mode': 'train'
        }
        torch.save(state, f'../save/proposedModel7.pt')