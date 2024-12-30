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

def getID(path):
    fname = path.split("/")[-1]
    subject = fname.split("--")[0]
    return subject

def getDate(path):
    fname = path.split("/")[-1]
    date = fname.split("--")[1]
    return date

adni_labels = pd.read_csv("./src/data/DXSUM_PDXCONV.csv")
adni_labels = {row.PTID: row.DIAGNOSIS for row in adni_labels.itertuples()}
adni_labels = {p: int(l - 1) for p, l in adni_labels.items() if l == l} # 0: CN, 1: MCI, 2: AD

test_dataset = pickle.load(open("./src/data/testDataset.pkl", "rb"))
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
    mri = resize_array(mri, real.shape[0])

    # Axial
    for i in range(real.shape[2] // 2, (real.shape[2] // 2) + 1):
        slice_mri = mri[:, :, i]
        slice_real = real[:, :, i]
        slice_synthetic = synthetic[:, :, i]
        combined_slice = np.hstack((slice_mri, slice_real, slice_synthetic))
        if combined_slice.max() > 0.05:
            plt.imsave(
                os.path.join(path_prefix, f"Axial_{key}_{i}.png"),
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
        generated_pet = model.generate(sample_contexts.to(device))
    for j in range(sample_contexts.size(0)):
        fpath = test_dataset[i+j][1]
        subject = getID(fpath)
        date = getDate(fpath)
        ad_status = adni_labels.get(subject, "Unknown")
        real_pet = tensor_to_numpy(real_images[j].cpu())
        real_mri = resize_array(sample_contexts[j].cpu().clone().numpy().transpose((1, 2, 0)), real_pet.shape[0])
        real_mri = align_mri(real_mri, real_pet)
        generated_pet = tensor_to_numpy(generated_pet[j].cpu())
        save_slice_plots(real_mri, real_pet, generated_pet, f"./src/results/case_study_samples/", f"{subject}_{date}_{ad_status}")
