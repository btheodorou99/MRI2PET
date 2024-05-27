import os
import sys
import ants
import numpy as np
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()

def process_file(pet_dir, niix_file, output_dir):
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

    img = ants.resample_image(img, (config.pet_image_dim, config.pet_image_dim, config.n_pet_channels), use_voxels=True, interp_type=3)
    img = ants.utils.convert_nibabel.to_nibabel(img)
    nib.save(img, os.path.join(output_dir, niix_file))

if __name__ == "__main__":
    pet_dir = sys.argv[1]
    niix_file = sys.argv[2]
    output_dir = sys.argv[3]
    process_file(pet_dir, niix_file, output_dir)