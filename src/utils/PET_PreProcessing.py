import os
import ants
import pickle
import random
import numpy as np
from tqdm import tqdm
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
output_dir = "/data/CARD_AA/data/ADNI/PET_Nifti_PreProcessed/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# {subject_id}--{date}--{some_ids}.nii
single_files = []
problem_files = []
allFiles = os.listdir(pet_dir)
allFiles = [f for f in allFiles if f.endswith('.nii') and not os.path.exists(os.path.join(output_dir, f))]
random.shuffle(allFiles)
for niix_file in tqdm(allFiles):
    print(niix_file)
    try:
        subject_id, date, some_ids = niix_file[:-4].split('--')
        img = ants.image_read(os.path.join(pet_dir, niix_file))
        if len(img.shape) == 4:
            num_time_points = img.shape[3]
            
            # Register to First on Initial Pass
            reference_frame = ants.from_numpy(img[:, :, :, 0])
            initial_registered_frames = [reference_frame.numpy()]
            for i in range(1, num_time_points):
                moving_frame = img[:, :, :, i]
                registration = ants.registration(fixed=reference_frame, moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
                initial_registered_frames.append(registration['warpedmovout'].numpy())

            # Register to Mean on Second Pass
            reference_frame = ants.from_numpy(np.mean(np.array(initial_registered_frames), axis=0))
            final_registered_frames = []
            for i in range(0, num_time_points):
                moving_frame = initial_registered_frames[i]
                registration = ants.registration(fixed=reference_frame, moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
                final_registered_frames.append(registration['warpedmovout'].numpy())

            img = ants.from_numpy(np.mean(np.array(final_registered_frames), axis=0))

        # Disregard images that are just a single slice
        if img.shape[2] == 1:
            single_files.append(niix_file)
            continue

        img = ants.resample_image(img, (config.pet_image_dim, config.pet_image_dim, config.n_pet_channels), use_voxels=True, interp_type=3)
        img = ants.utils.convert_nibabel.to_nibabel(img)
        nib.save(img, os.path.join(output_dir, niix_file))
    except:
        problem_files.append(niix_file)

pickle.dump(single_files, open('./src/data/single_nii.pkl', 'wb'))
pickle.dump(problem_files, open('./src/data/problem_nii.pkl', 'wb'))