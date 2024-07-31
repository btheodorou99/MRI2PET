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
plotDir1 = "/data/theodoroubp/MRI2PET/results/data_exploration/"
plotDir2 = "/data/theodoroubp/MRI2PET/results/data_exploration2/"
pairs = pickle.load(open('./src/data/pet_mri_pairs.pkl', 'rb'))
pairs = [(mri_path.replace('.npy', '.nii.gz'), pet_path.split('/')[-1]) for mri_path, pet_path in pairs]
random.shuffle(pairs)
pet_template_path = "./src/data/petTemplate.nii"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if os.path.exists(pet_template_path):
    pet_template = ants.image_read(pet_template_path)
else:
    pet_template = np.zeros((config.pet_image_dim, config.pet_image_dim, config.n_pet_channels))
    numPets = 0
    for niix_file in tqdm(os.listdir(pet_dir)):
        if not niix_file.endswith('.nii'):
            continue
        img = ants.image_read(os.path.join(pet_dir, niix_file))
        pet_template += img.numpy()
        numPets += 1
    pet_template /= numPets
    pet_template = ants.from_numpy(pet_template)
    pet_nii = ants.utils.convert_nibabel.to_nibabel(pet_template)
    nib.save(pet_nii, pet_template_path)

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
    img = ants.registration(fixed=pet_template, moving=img, type_of_transform='Rigid')['warpedmovout']        
    ants.plot(
            ants.from_numpy(img.numpy().transpose(2, 0, 1)),
            nslices=9,
            title=str(img.shape[0]),
            filename=f"{plotDir1}{pet_file.replace('.npy', '.png')}",
    )
    img = ants.registration(fixed=mri_img, moving=img, type_of_transform='Affine')['warpedmovout']        
    ants.plot(
            ants.from_numpy(img.numpy().transpose(2, 0, 1)),
            nslices=9,
            title=str(img.shape[0]),
            filename=f"{plotDir1}{pet_file.replace('.npy', '.png')}",
    )
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())

    # Save processed data
    np.save(output_filename, data)