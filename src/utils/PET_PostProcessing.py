import os
import ants
import pickle
import random
import numpy as np
from tqdm import tqdm
import nibabel as nib
from src.config import MRI2PETConfig

config = MRI2PETConfig()
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti_PreProcessed/"
output_dir = "/data/CARD_AA/data/ADNI/PET/"
pairs = pickle.load(open('./src/data/pet_mri_pairs.pkl', 'rb'))
pairs = [(mri_path.replace('.npy', '.nii.gz'), pet_path.split('/')[-1]) for mri_path, pet_path in pairs]
random.shuffle(pairs)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Perform Registrations to PET Mean and Subject MRI
for mri_niix, pet_file in tqdm(pairs):
    pet_niix = os.path.join(pet_dir, pet_file).replace('.npy', '.nii')
    npy_filename = pet_file.replace('.nii', '.npy')
    output_filename = os.path.join(output_dir, npy_filename)
    if os.path.exists(output_filename):
        continue

    img = ants.image_read(pet_niix)
    mri_img = ants.image_read(mri_niix)
    mri_img = ants.resample_image(img, (config.pet_image_dim, config.pet_image_dim, config.n_pet_channels), use_voxels=True, interp_type=3)
    img = ants.registration(fixed=mri_img, moving=img, type_of_transform='Affine')['warpedmovout']        
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())

    # Save processed data
    np.save(output_filename, data)