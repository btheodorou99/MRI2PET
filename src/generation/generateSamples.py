import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..config import MRI2PETConfig
from ..models.ganModel import Generator
from ..models.diffusionModel import DiffusionModel
from ..models.diffusionModel3D import DiffusionModel as DiffusionModel3D

SEED = 4
cudaNum = 0
NUM_SAMPLES = 5
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

def tensor_to_numpy(tensor):
    """Convert a torch tensor to a numpy array."""
    # Convert to a numpy array
    image = tensor.cpu().clone()
    image = (image + 1) / 2.0
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    return image

def save_slice_plots(array, path_prefix):
    # Array: (Height, Width, Slices)
    # Directions: Transversal, Sagittal, Coronal
    
    # Transversal
    for i in range(array.shape[2]):
        slice = array[:, :, i]
        if i % 5 == 0 and slice.max() > 0:
            plt.imsave(f'{path_prefix}_Transversal{i}.png', slice, cmap='gray')
        
    # Sagittal
    for i in range(array.shape[1]):
        slice = array[:, i, :]
        if i % 5 == 0 and slice.max() > 0: 
            plt.imsave(f'{path_prefix}_Sagittal{i}.png', slice, cmap='gray')
        
    # Coronal
    for i in range(array.shape[0]):
        slice = array[i, :, :]
        if i % 5 == 0 and slice.max() > 0:
            plt.imsave(f'{path_prefix}_Coronal{i}.png', slice, cmap='gray')
        

# Generate Samples
sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
sample_data = [test_dataset[i] for i in sampleIdx]
sample_contexts, real_images = get_batch(sample_data, 0, NUM_SAMPLES)


for i in range(NUM_SAMPLES):
    real_image = tensor_to_numpy(real_images[i].cpu())
    save_slice_plots(real_image, f'./src/results/image_samples/realImage_{i}')

model_keys = [
    'baseGAN',
    'baseDiffusion',
    'baseDiffusion3D',
    # 'noisyPretrainedDiffusion',
    # 'noisyPretrainedGAN',
    # 'selfPretrainedDiffusion',
    # 'selfPretrainedGAN',
    'stylePretrainedDiffusion',
    # 'stylePretrainedGAN',
    # 'mri2pet',
]

for k in tqdm(model_keys):
    print(k)
    if 'GAN' in k:
        model = Generator(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['generator'])
        model = model.to(device)
        model.eval()
    elif '3D' in k:
        model = DiffusionModel3D(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()
    else:
        model = DiffusionModel(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()

    with torch.no_grad():
        sample_images = model.generate(sample_contexts)

    for i in range(NUM_SAMPLES):
        sample_image = tensor_to_numpy(sample_images[i].cpu())
        save_slice_plots(sample_image, f'./src/results/image_samples/{k}_{i}')