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

mri_dataset = [m for m in pickle.load(open('./src/data/mriDataset.pkl', 'rb')) if '/ADNI/' in m]
global_mean = pickle.load(open("./src/data/globalMean.pkl", "rb"))
global_std = pickle.load(open("./src/data/globalStd.pkl", "rb"))

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, device=device)
    for i, p in enumerate(image_paths):
        batch_context[i] = load_image(p)
        
    return batch_context, image_paths
  
def save_image(tensor, path):
    """Save a torch tensor as an image."""
    # Convert the tensor to a PIL image and save
    image = tensor.cpu().clone()
    image = (image * global_std) + global_mean
    image = (image - image.min()) / (image.max() - image.min())
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    np.save(path, image)
        
k = 'mri2pet_pScale_tweaked'
os.makedirs(f'/data/theodoroubp/MRI2PET/results/downstream_dataset/', exist_ok=True)
model = DiffusionModel(config)
model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
model = model.to(device)
model.eval()
batch_size = config.batch_size // 3

paired_dataset = []
with torch.no_grad():
    for i in tqdm(range(0, len(mri_dataset), batch_size), leave=False):
        sample_contexts, mri_paths = get_batch(mri_dataset, i, batch_size)
        sample_images = model.generate(sample_contexts)
        for j in range(sample_images.size(0)):
            pet_path = f'/data/theodoroubp/MRI2PET/results/downstream_dataset/syntheticImage_{i+j}.npy'
            save_image(sample_images[j], pet_path)
            paired_dataset.append((mri_paths[j], pet_path))

pickle.dump(paired_dataset, open(f'./src/data/syntheticDataset.pkl', 'wb'))