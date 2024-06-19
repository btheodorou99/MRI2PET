import os
import ants
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from ..config import MRI2PETConfig

SEED = 4
cudaNum = 0
NUM_SAMPLES = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

output_dir = "/data/CARD_AA/data/ADNI/PET/"
sample_paths = random.choices(os.listdir(output_dir), k=NUM_SAMPLES)

def tensor_to_numpy(tensor):
    """Convert a torch tensor to a numpy array."""
    # Convert to a numpy array
    image = tensor.cpu().clone()
    image = (image + 1) / 2.0
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    return image

def save_slice_plots(array, dir, path_suffix):
    # Array: (Height, Width, Slices)
    # Directions: Axial, Sagittal, Coronal
    
    # Axial
    for i in range(array.shape[2]):
        slice = array[:, :, i]
        if i % 10 == 0 and slice.max():
            plt.imsave(f'{dir}/Axial{i}_{path_suffix}.png', slice, cmap='gray')
        
    # Sagittal
    for i in range(array.shape[1]):
        slice = array[:, i, :]
        if i % 10 == 0 and slice.max():
            plt.imsave(f'{dir}/Sagittal{i}_{path_suffix}.png', slice, cmap='gray')
        
    # Coronal
    for i in range(array.shape[0]):
        slice = array[i, :, :]
        if i % 10 == 0 and slice.max():
            plt.imsave(f'{dir}/Coronal{i}_{path_suffix}.png', slice, cmap='gray')
        
for i in range(NUM_SAMPLES):
    real_image = ants.image_read(f'{output_dir}{sample_paths[i]}').numpy()
    if len(real_image.shape) == 4:
        real_image = real_image[:, :, :, 0]
    real_image = (real_image - real_image.min()) / (real_image.max() - real_image.min())
    print(real_image.shape)
    save_slice_plots(real_image, f'./src/results/pet_samples', i)