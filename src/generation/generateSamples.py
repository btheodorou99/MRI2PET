import ants
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from ..config import MRI2PETConfig
from ..models.ganModel import Generator
from ..models.diffusionModel import DiffusionModel

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

def tensor_to_numpy(tensor, isGan=False):
    """Convert a torch tensor to a numpy array."""
    # Convert to a numpy array
    image = tensor.cpu().clone()
    if isGan:
        image = (image + 1.0) / 2.0
    else:
        image = (image * global_std) + global_mean
    image = (image - image.min()) / (image.max() - image.min())
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

def align_mri(mri, pet):
    mri = ants.from_numpy(mri)
    pet = ants.from_numpy(pet)
    mri = ants.registration(fixed=pet, moving=mri, type_of_transform='Affine')['warpedmovout']
    mri = mri.numpy()
    mri = (mri - mri.min()) / (mri.max() - mri.min())
    return mri

def expand_slices(array, target_shape):
    zoom_factors = (
        1,
        1,
        target_shape / array.shape[2]
    )
    return zoom(array, zoom_factors)

def save_slice_plots(array, path_prefix):
    # Array: (Height, Width, Slices)
    # Directions: Axial, Sagittal, Coronal
    
    # Axial
    for i in range(5,20):
        slice = array[:, :, i]
        plt.imsave(f'{path_prefix}_Axial{i}.png', slice, cmap='gray')
        
    # # For other views, we need to resize to get a square
    # array = expand_slices(array, array.shape[0])

    # # Sagittal
    # for i in range(array.shape[1]):
    #     slice = array[:, i, :]
    #     if i % 5 == 0 and slice.max() > 0.05: 
    #         plt.imsave(f'{path_prefix}_Sagittal{i}.png', slice, cmap='gray')
        
    # # Coronal
    # for i in range(array.shape[0]):
    #     slice = array[i, :, :]
    #     if i % 5 == 0 and slice.max() > 0.05:
    #         plt.imsave(f'{path_prefix}_Coronal{i}.png', slice, cmap='gray')
        

# Generate Samples
sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
sample_data = [test_dataset[i] for i in sampleIdx]
sample_contexts, real_images = get_batch(sample_data, 0, NUM_SAMPLES)


for i in range(NUM_SAMPLES):
    real_pet = tensor_to_numpy(real_images[i].cpu())
    real_mri = resize_array(sample_contexts[i].cpu().clone().numpy().transpose((1, 2, 0)), real_pet.shape[0])
    aligned_mri = align_mri(np.copy(real_mri), real_pet)
    save_slice_plots(real_mri, f'./src/results/image_samples/realMRI_{i}')
    save_slice_plots(aligned_mri, f'./src/results/image_samples/realMRI_Aligned_{i}')
    save_slice_plots(real_pet, f'./src/results/image_samples/realPET_{i}')

model_keys = [
    'baseDiffusion',
    'baseGAN',
    'mri2pet',
    'mri2pet_pScale',
    'mri2pet_sameOptimizer',
    'mri2pet_smallerLoss',
    'mri2pet_noPretrain',
    'mri2pet_noLoss',
    'mri2pet_noLoss_pScale',
    'selfPretrainedDiffusion',
    'selfPretrainedDiffusion_pScale',
    'noisyPretrainedDiffusion',
    'noisyPretrainedDiffusion_pScale',
    'baseDiffusion_tweaked',
    'baseGAN_tweaked',
    'mri2pet_pScale_tweaked',
    'mri2pet_noPretrain_tweaked',
    'mri2pet_noLoss_pScale_tweaked',
    'selfPretrainedDiffusion_pScale_tweaked',
    # 'noisyPretrainedDiffusion_pScale_tweaked',
]

for k in tqdm(model_keys):
    print(k)
    isGan = 'GAN' in k
    if 'GAN' in k:
        model = Generator(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['generator'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size
    else:
        model = DiffusionModel(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size // 5

    with torch.no_grad():
        sample_images = model.generate(sample_contexts)

    for i in range(NUM_SAMPLES):
        sample_image = tensor_to_numpy(sample_images[i].cpu(), isGan)
        save_slice_plots(sample_image, f'./src/results/image_samples/{k}_{i}')