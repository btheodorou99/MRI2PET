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
NUM_SAMPLES = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))

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

model_keys = [
    'baseDiffusion',
    'baseDiffusionGradientClip',
    'baseDiffusionNoiseRampup',
    # 'noisyPretrainedDiffusion',
    # 'selfPretrainedDiffusion',
    # 'stylePretrainedDiffusion',
    # 'mri2pet',
]

for k in tqdm(model_keys):
    print(k)
    model = DiffusionModel(config).to(device)
    model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location=torch.device(device))['model'])
    model.eval()

    total_nll = 0.0
    total_pixels = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), config.batch_size), leave=False):
            batch_context, batch_images = get_batch(test_dataset, i, config.batch_size)
            nll = model.calculate_nll(batch_context, batch_images)
            total_nll += nll
            total_pixels += batch_images.numel()

    
    nll = total_nll / total_pixels
    bits_per_dim = total_nll / (np.log(2))
    print('{k} NLL:', nll, 'BPD:', bits_per_dim)
    pickle.dump(nll, open(f'./src/results/quantitative_evaluations/{k}_nll.pkl', 'wb'))
    pickle.dump(bits_per_dim, open(f'./src/results/quantitative_evaluations/{k}_bits_per_dim.pkl', 'wb'))