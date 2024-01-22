import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from ..config import MRI2PETConfig

input_mri_dir = "../data/MRI/"
output_mri_dir = "../data/MRI_Processed/"
input_pet_dir = "../data/PET/"
output_pet_dir = "../data/PET_Processed/"

os.makedirs(output_mri_dir, exist_ok=True)
os.makedirs(output_pet_dir, exist_ok=True)

config = MRI2PETConfig()

MRI_RESOLUTION = config.mri_image_dim
MRI_SLICES = config.n_mri_channels
PET_RESOLUTION = config.pet_image_dim
PET_SLICES = config.n_pet_channels

def standardize_shape(img, target_resolution, target_slices):
    # Current shape
    current_height, current_width, current_slices = img.shape

    # Resizing each slice to the target resolution
    height_zoom = target_resolution / current_height
    width_zoom = target_resolution / current_width
    resized_img = zoom(img, (height_zoom, width_zoom, 1), order=5)

    # Adjusting the number of slices
    slice_zoom = target_slices / current_slices
    standardized_img = zoom(resized_img, (1, 1, slice_zoom), order=5)

    return standardized_img

mri_paths = pickle.load(open('../data/mriDataset.pkl', 'rb'))
pet_paths = [p for (m, p) in pickle.load(open('../data/pet_mri_pairs.pkl', 'rb'))]

for mri_path in tqdm(mri_paths):
    mri = np.load(f'{input_mri_dir}{mri_path}')
    standardized_mri = standardize_shape(mri, MRI_RESOLUTION, MRI_SLICES)
    np.save(f'{output_mri_dir}{mri_path}', standardized_mri)
    
for pet_path in tqdm(pet_paths):
    pet = np.load(f'{input_pet_dir}{pet_path}')
    standardized_pet = standardize_shape(pet, PET_RESOLUTION, PET_SLICES)
    np.save(f'{output_pet_dir}{pet_path}', standardized_pet)