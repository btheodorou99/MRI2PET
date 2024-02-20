import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from ..config import MRI2PETConfig

input_mri_dir = "./src/data/MRI_Processed/"
output_mri_dir = "./src/data/MRI_Pretrain/"
pet_dir = "./src/data/PET_Processed/"

os.makedirs(output_mri_dir, exist_ok=True)

config = MRI2PETConfig()

MRI_HEIGHT = config.mri_image_dim
MRI_WIDTH = config.mri_image_dim
MRI_SLICES = config.n_mri_channels
PET_HEIGHT = config.pet_image_dim
PET_WIDTH = config.pet_image_dim
PET_SLICES = config.n_pet_channels

height_zoom = PET_HEIGHT / MRI_HEIGHT
width_zoom = PET_WIDTH / MRI_WIDTH
slice_zoom = PET_SLICES / MRI_SLICES

def scale_image(mri_image):
    # Resizing each slice to the target resolution
    resized_mri = zoom(mri_image, (height_zoom, width_zoom, 1), order=5)
    resized_mri = zoom(resized_mri, (1, 1, slice_zoom), order=5)
    resized_mri = resized_mri.clip(0, 1)
    return resized_mri

mri_paths = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
for mri_path in tqdm(mri_paths):
    mri = np.load(f'{input_mri_dir}{mri_path}')
    standardized_mri = scale_image(mri)
    np.save(f'{output_mri_dir}{mri_path}', standardized_mri)