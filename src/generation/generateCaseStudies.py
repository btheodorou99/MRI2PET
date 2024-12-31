import os
import ants
import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
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

interesting_subjects = {
    ('051_S_1072', '2012-02-29'): 'MCI_WeirdShape',
    ('002_S_4521', '2016-03-29'): 'AD_BetterResolutionwSulkusOnBottom',
    ('094_S_2216', '2015-03-17'): 'AD_LongitudinalProgression',
    ('135_S_4722', '2012-06-05'): 'MCI_LongitudinalProgression',
    ('114_S_6524', '2020-12-08'): 'CN_Atrophy',
    ('020_S_5203', '2015-05-27'): 'CN_ImageQuality',
    ('129_S_6784', '2019-09-25'): 'AD_LongitudinalProgression',
    ('129_S_6784', '2021-08-20'): 'AD_LongitudinalProgression',
    ('041_S_1010', '2010-12-10'): 'AD_GyriBetterAndWorse',
    ('037_S_5162', '2013-05-30'): 'AD_SulksTopAndBottom',
    ('021_S_6914', '2021-03-24'): 'CN_ThingOnLeft',
    ('126_S_5214', '2013-08-15'): 'MCI_Atrophy',
    ('036_S_4562', '2012-04-12'): 'MCI_Gyri',
    ('037_S_4030', '2015-05-27'): 'AD_VerySimilar',
    ('135_S_4722', '2021-06-18'): 'MCI_LongitudinalProgression',
    ('002_S_5178', '2017-06-05'): 'CN_BrightGyriAroundGap',
    ('005_S_0610', '2014-12-09'): 'CN_ImageProcess',
    ('035_S_4785', '2019-12-10'): 'MCI_GyriClear',
    ('126_S_4891', '2018-08-22'): 'AD_Atrophy',
    ('094_S_2216', '2013-03-12'): 'AD_LongitudinalProgression',
    ('021_S_4744', '2012-06-07'): 'MCI_GyriRightSide',
    ('099_S_6025', '2021-11-04'): 'CN_DeepSulkus'
}    

def getID(path):
    fname = path.split("/")[-1]
    subject = fname.split("--")[0]
    return subject

def getDate(path):
    fname = path.split("/")[-1]
    date = fname.split("--")[1]
    return date

test_dataset = pickle.load(open("./src/data/testDataset.pkl", "rb"))
test_dataset = [x for x in test_dataset if (getID(x[1]), getDate(x[1])) in interesting_subjects]
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

def tensor_to_numpy(tensor):
    """Convert a torch tensor to a numpy array."""
    # Convert to a numpy array
    image = tensor.cpu().clone()
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


def save_slice_plots(mri, real, synthetic, path_prefix, key):
    # Array: (Height, Width, Slices)
    # Directions: Axial, Sagittal, Coronal

    # Axial
    for i in range(real.shape[2] // 3, real.shape[2] * 2 // 3):
        slice_mri = mri[:, :, i]
        slice_real = real[:, :, i]
        slice_synthetic = synthetic[:, :, i]
        combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
        if combined_slice.max() > 0.05:
            plt.imsave(
                os.path.join(path_prefix, f"{key}_{i}.png"),
                combined_slice,
                cmap="gray",
            )

    # # For other views, we need to resize to get a square
    # mri = expand_slices(mri, real.shape[0])
    # real = expand_slices(real, real.shape[0])
    # synthetic = expand_slices(synthetic, real.shape[0])

    # # Sagittal
    # for i in range(real.shape[1]):
    #     slice_mri = mri[:, i, :]
    #     slice_real = real[:, i, :]
    #     slice_synthetic = synthetic[:, i, :]
    #     combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
    #     if combined_slice.max() > 0.05:
    #         plt.imsave(
    #             os.path.join(path_prefix, f"Sagittal_{key}_{i}.png"),
    #             combined_slice,
    #             cmap="gray",
    #         )

    # # Coronal
    # for i in range(real.shape[0]):
    #     slice_mri = mri[i, :, :]
    #     slice_real = real[i, :, :]
    #     slice_synthetic = synthetic[i, :, :]
    #     combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
    #     if combined_slice.max() > 0.05:
    #         plt.imsave(
    #             os.path.join(path_prefix, f"Coronal_{key}_{i}.png"),
    #             combined_slice,
    #             cmap="gray",
    #         )

k = "mri2pet"
model = DiffusionModel(config)
model.load_state_dict(torch.load(f"./src/save/{k}.pt", map_location="cpu")["model"])
model = model.to(device)
model.eval()
config.batch_size = config.batch_size // 5
os.makedirs("./src/results/case_study_samples", exist_ok=True)
for i in tqdm(range(0, len(test_dataset), config.batch_size)):
    sample_contexts, real_images = get_batch(test_dataset, i, config.batch_size)
    with torch.no_grad():
        generated_batch = model.generate(sample_contexts.to(device)).cpu()
    for j in range(sample_contexts.size(0)):
        fpath = test_dataset[i+j][1]
        subject = getID(fpath)
        date = getDate(fpath)
        note = interesting_subjects[(subject, date)]
        real_pet = tensor_to_numpy(real_images[j].cpu())
        real_mri = resize_array(sample_contexts[j].cpu().clone().numpy().transpose((1, 2, 0)), real_pet.shape[0])
        real_mri = align_mri(real_mri, real_pet)
        generated_pet = tensor_to_numpy(generated_batch[j])
        save_slice_plots(real_mri, real_pet, generated_pet, f"./src/results/case_study_samples/", f"{subject}_{date}_{note}")
