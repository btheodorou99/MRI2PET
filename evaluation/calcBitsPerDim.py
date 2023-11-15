import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from ..config import MRI2PETConfig
from ..models.diffusionModel import DiffusionModel
from ..models.ganModel import Generator

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

test_dataset = pickle.load(open('../data/testDataset.pkl', 'rb'))
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

def load_image(image_path, is_mri=True):
    with Image.open(f'{config.image_dir}/{image_path}.jpg') as img:
        return image_transform_mri(img) if is_mri else image_transform_pet(img)

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float, device=device)
    batch_image = torch.zeros(bs, config.n_channels, config.pet_image_dim, config.pet_image_dim, dtype=torch.float, device=device)
    for i, (m, p) in enumerate(image_paths):
        batch_context[i] = load_image(m, is_mri=True)
        batch_image[i] = load_image(p, is_mri=False)
        
    return batch_context, batch_image

def tensor_to_image(tensor):
    # First de-normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert to PIL image
    img = transforms.ToPILImage()(tensor)
    return img

model_keys = [
    'baseDiffusion',
    'baseGAN',
    'noisyPretrainedDiffusion',
    'noisyPretrainedGAN',
    'selfPretrainedDiffusion',
    'selfPretrainedGAN',
    'stylePretrainedDiffusion',
    'stylePretrainedGAN',
    'mri2pet',
]

for k in tqdm(model_keys):
    print(k)
    model = DiffusionModel(config).to(device)
    model.load_state_dict(torch.load(f'../save/{k}.pt', map_location=torch.device(device))['model'])
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
    pickle.dump(nll, open(f'../results/quantitative_evaluations/{k}_nll.pkl', 'wb'))
    pickle.dump(bits_per_dim, open(f'../results/quantitative_evaluations/{k}_bits_per_dim.pkl', 'wb'))