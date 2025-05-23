import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from ..config import MRI2PETConfig
from ..models.diffusionModel import DiffusionModel

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

test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))
global_mean = pickle.load(open("./src/data/globalMean.pkl", "rb"))
global_std = pickle.load(open("./src/data/globalStd.pkl", "rb"))

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if not is_mri:
        img = (img - global_mean) / global_std
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

model_keys = [
    'baseDiffusion',
    # 'baseGAN',
    'mri2pet',
    'mri2pet_noPretrain',
    'mri2pet_noLoss',
    'selfPretrainedDiffusion',
    'noisyPretrainedDiffusion',
]

config.batch_size = config.batch_size // 5
for k in tqdm(model_keys):
    print(k)
    model = DiffusionModel(config).to(device)
    model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location=torch.device(device))['model'])
    model.eval()

    nlls = []
    pixels = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), config.batch_size), leave=False):
            batch_context, batch_images = get_batch(test_dataset, i, config.batch_size)
            nll = model.calculate_nll(batch_context, batch_images)
            nlls += nll
            pixels += [batch_images[i].numel() for i in range(batch_images.size(0))]

    nlls = np.array(nlls)
    pixels = np.array(pixels)
    bpds = []
    for i in range(config.n_bootstrap):
        bs_indices = np.random.choice(np.arange(len(nlls)), replace=True)
        bs_nlls = nlls[bs_indices]
        bs_pixels = pixels[bs_indices]
        total_nll = np.sum(bs_nlls)
        total_pixels = np.sum(bs_pixels)
        nll = total_nll / total_pixels
        bits_per_dim = nll / (np.log(2))
        bpds.append(bits_per_dim)
    
    bits_per_dim = (np.mean(bpds), np.std(bpds) / np.sqrt(config.n_bootstrap))
    print(f'BPD: {bits_per_dim[0]:.3f} \\pm {bits_per_dim[1]:.3f}')
    pickle.dump(bits_per_dim, open(f'./src/results/quantitative_evaluations/{k}_bits_per_dim.pkl', 'wb'))