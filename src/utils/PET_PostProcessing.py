import os
import ants
import pickle
import numpy as np
from tqdm import tqdm
import nibabel as nib
from src.config import MRI2PETConfig

data_dict = {}
config = MRI2PETConfig()
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
mri_dir = "/data/CARD_AA/data/ADNI/MRI_Nifti/"
output_dir = "/data/CARD_AA/data/ADNI/PET/"
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
petToMri = pickle.load(open('./src/data/petToMri.pkl', 'rb'))
for niix_file in tqdm(os.listdir(pet_dir)):
    if not niix_file.endswith('.nii'):
        continue

    subject_id, date, some_ids = niix_file[:-4].split('--')  
    npy_filename = f"{subject_id}--{date}--{some_ids}.npy"
    output_filename = os.path.join(output_dir, npy_filename)
    if os.path.exists(output_filename):
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        data_dict[subject_id][date] = {'shape': np.load(output_filename).shape, 'filename': npy_filename}
        continue

    img = ants.image_read(os.path.join(pet_dir, niix_file))
    mri_img = ants.image_read(os.path.join(mri_dir, petToMri[subject_id][date]))
    img = ants.registration(fixed=pet_template, moving=img, type_of_transform='Rigid')['warpedmovout']        
    img = ants.registration(fixed=mri_img, moving=img, type_of_transform='Rigid')['warpedmovout']        
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())

    # Save processed data
    np.save(output_filename, data)

    # Update dictionary
    if subject_id not in data_dict:
        data_dict[subject_id] = {}
    data_dict[subject_id][date] = {'shape': data.shape, 'filename': npy_filename}

pickle.dump(data_dict, open('./src/data/pet_dict.pkl', 'wb'))