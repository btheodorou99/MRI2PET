import os
import sys
import ants
import antspynet
import numpy as np
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()

template = ants.image_read('/home/theodoroubp/MRI2PET/src/data/petTemplate.nii')

def process_file(pet_dir, niix_file, output_dir):
    img = ants.image_read(os.path.join(pet_dir, niix_file))
    if len(img.shape) == 4:
        num_time_points = img.shape[3]
        
        # Register to First on Initial Pass
        reference_frame = ants.slice_image(img, axis=3, idx=0)
        initial_registered_frames = [reference_frame]
        for i in range(1, num_time_points):
            registration = ants.registration(fixed=reference_frame, moving=ants.slice_image(img, axis=3, idx=i), type_of_transform='Rigid')
            initial_registered_frames.append(registration['warpedmovout'])

        # Register to Mean on Second Pass
        reference_frame = ants.from_numpy(np.mean(np.array([f.numpy() for f in initial_registered_frames]), axis=0), origin=reference_frame.origin, spacing=reference_frame.spacing, direction=reference_frame.direction)
        final_registered_frames = []
        for i in range(0, num_time_points):
            moving_frame = initial_registered_frames[i]
            registration = ants.registration(fixed=reference_frame, moving=moving_frame, type_of_transform='Rigid')
            final_registered_frames.append(registration['warpedmovout'].numpy())

        img = ants.from_numpy(np.mean(np.array(final_registered_frames), axis=0), origin=reference_frame.origin, spacing=reference_frame.spacing, direction=reference_frame.direction)

    # Register to PET template
    img = ants.registration(fixed=template, moving=img, type_of_transform="Affine")['warpedmovout']

    # Strip the skull
    seg = antspynet.brain_extraction(img, modality="t1")
    img = ants.from_numpy(img.numpy() * seg.numpy(), origin=img.origin, spacing=img.spacing, direction=img.direction)
    
    # Save the image
    img = ants.to_nibabel(img)
    nib.save(img, os.path.join(output_dir, niix_file))

if __name__ == "__main__":
    pet_dir = sys.argv[1]
    niix_file = sys.argv[2]
    output_dir = sys.argv[3]
    process_file(pet_dir, niix_file, output_dir)