import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig
from ..models.diffusionModel import DiffusionModel

SEED = 4
cudaNum = 0
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

model = DiffusionModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(f"./src/save/baseDiffusionGradientClip.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/baseDiffusionGradientClip.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    train_losses = []
    model.train()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        optimizer.zero_grad()
        loss, _ = model(batch_context, batch_images, gen_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.cpu().detach().item())
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for v_i in range(0, len(val_dataset), config.batch_size):
            batch_context, batch_images = get_batch(val_dataset, v_i, config.batch_size)                 
            val_loss, _ = model(batch_context, batch_images, gen_loss=True)
            val_losses.append((val_loss).cpu().detach().item())
        
        cur_train_loss = np.mean(train_losses)
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Training Loss: %.7f, Validation Loss:%.7f"%(e, cur_train_loss, cur_val_loss))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f'./src/save/baseDiffusionGradientClip.pt')