import os
import ants
import numpy as np
from tqdm import tqdm
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
output_dir = "/data/CARD_AA/data/ADNI/PET_Nifti_PreProcessed/"
# template = ants.image_read('./src/data/petTemplate.nii')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# nii_filename = f"{subject_id}--{date}--{some_ids}.nii"
for niix_file in tqdm(os.listdir(pet_dir)):
    if not niix_file.endswith('.nii') or os.path.exists(os.path.join(output_dir, niix_file)):
        continue

    subject_id, date, some_ids = niix_file[:-4].split('--')
    img = ants.image_read(os.path.join(pet_dir, niix_file))
    if len(img.shape) == 4:
        num_time_points = img.shape[3]
        
        # Register to First on Initial Pass
        reference_frame = ants.from_numpy(img[:, :, :, 0])
        initial_registered_frames = [reference_frame]
        for i in range(1, num_time_points):
            moving_frame = img[:, :, :, i]
            registration = ants.registration(fixed=reference_frame, moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
            initial_registered_frames.append(registration['warpedmovout'].numpy())

        # Register to Mean on Second Pass
        reference_frame = ants.from_numpy(np.mean(np.array(initial_registered_frames), axis=0))
        final_registered_frames = []
        for i in range(0, num_time_points):
            moving_frame = final_registered_frames[i]
            registration = ants.registration(fixed=reference_frame, moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
            final_registered_frames.append(registration['warpedmovout'].numpy())

        img = ants.from_numpy(np.mean(np.array(final_registered_frames), axis=0))

    # Disregard images that are just a single slice
    if img.shape[2] == 1:
        continue

    img = ants.resample_image(img, (config.pet_image_dim, config.pet_image_dim, config.n_pet_channels), use_voxels=True, interp_type=3)
    # img = ants.registration(fixed=template, moving=img, type_of_transform='SyN')['warpedmovout']
    img = ants.utils.convert_nibabel.to_nibabel(img)
    nib.save(img, os.path.join(output_dir, niix_file))