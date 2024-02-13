import os
import ants
import pickle
import subprocess
import numpy as np
from tqdm import tqdm

data_dict = {}
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
mri_dir = "/data/CARD_AA/data/ADNI/MRI_Nifti/"
output_dir = "/data/CARD_AA/data/ADNI/PET/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

petToMri = pickle.load(open('../data/petToMri.pkl', 'rb'))
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
    img = ants.registration(fixed=mri_img, moving=img, type_of_transform='Rigid')['warpedmovout']        
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())

    # Save processed data
    np.save(output_filename, data)

    # Update dictionary
    if subject_id not in data_dict:
        data_dict[subject_id] = {}
    data_dict[subject_id][date] = {'shape': data.shape, 'filename': npy_filename}

pickle.dump(data_dict, open('../data/pet_dict.pkl', 'wb'))