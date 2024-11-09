import os
import torch
import random
import pickle
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from ..config import MRI2PETConfig
from ..models.diffusionModel import DiffusionModel

SEED = 1234
cudaNum = 0
NUM_SAMPLES = 100
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

def resize_array(array, target_shape):
    # Calculate the zoom factors for each dimension
    zoom_factors = (
        target_shape / array.shape[0],
        target_shape / array.shape[1],
        1 
    )
    return zoom(array, zoom_factors)

def expand_slices(array, target_shape):
    zoom_factors = (
        1,
        1,
        target_shape / array.shape[2]
    )
    return zoom(array, zoom_factors)

def save_slice_plots(mri, real, synthetic, path_prefix, num):
    # Array: (Height, Width, Slices)
    # Directions: Axial, Sagittal, Coronal
    mri = resize_array(mri, real.shape[0])

    # Axial
    for i in range(real.shape[2]):
        slice_mri = mri[:, :, i+20]
        slice_real = real[:, :, i]
        slice_synthetic = synthetic[:, :, i]
        combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
        if combined_slice.max() > 0.05:
            plt.imsave(os.path.join(path_prefix, f'Axial_{num}_{i}.png'), slice, cmap='gray')
        
    # For other views, we need to resize to get a square
    mri = expand_slices(mri, real.shape[0])
    real = expand_slices(real, real.shape[0])
    synthetic = expand_slices(synthetic, real.shape[0])
    
    # Sagittal
    for i in range(real.shape[1]):
        slice_mri = mri[:, i, :]
        slice_real = real[:, i, :]
        slice_synthetic = synthetic[:, i, :]
        combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
        if combined_slice.max() > 0.05:
            plt.imsave(f'{path_prefix}_Sagittal{i}.png', slice, cmap='gray')
        
    # Coronal
    for i in range(real.shape[0]):
        slice_mri = mri[i, :, :]
        slice_real = real[i, :, :]
        slice_synthetic = synthetic[i, :, :]
        combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
        if combined_slice.max() > 0.05:
            plt.imsave(f'{path_prefix}_Coronal{i}.png', slice, cmap='gray')
        

# Generate Samples
sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
sample_data = [test_dataset[i] for i in sampleIdx]
sample_contexts, real_images = get_batch(sample_data, 0, NUM_SAMPLES)

model_key = 'mri2pet_base_loss'
model = DiffusionModel(config)
model.load_state_dict(torch.load(f'./src/save/{model_key}.pt', map_location='cpu')['model'])
model = model.to(device)
model.eval()

sample_images = []
config.batch_size = config.batch_size // 3
with torch.no_grad():
    for i in range(0, NUM_SAMPLES, config.batch_size):
        batch_contexts = sample_contexts[i:i+config.batch_size]
        batch_images = model.generate(batch_contexts)
        sample_images.append(batch_images.cpu())

sample_images = torch.cat(sample_images, dim=0)

for i in range(NUM_SAMPLES):
    real_mri = sample_contexts[i].cpu().clone().numpy().transpose((1, 2, 0))
    real_pet = tensor_to_numpy(real_images[i].cpu())
    synthetic_pet = tensor_to_numpy(sample_images[i].cpu())
    save_slice_plots(real_mri, real_pet, synthetic_pet, f'./src/results/image_samples/', i)