import os
import numpy as np
from PIL import Image

# Variables
image_dir = '/Users/theodoroubp/Desktop/Samples2/'
model_keys = [
    'realMRI_Aligned', 
    'realPET', 
    'mri2pet_pScale_tweaked',
    'mri2pet_noLoss_pScale_tweaked',
    'mri2pet_noPretrain_tweaked',
    'baseDiffusion_tweaked',
    # 'selfPretrainedDiffusion_pScale_tweaked',
]
num_images = 5
slice_indices = [9, 12, 15]
image_size = 128
gap_size = 64

# Calculate the total width of the large image array
total_width = len(model_keys) * len(slice_indices) * image_size + (len(model_keys) - 1) * gap_size

# Initialize the large image array
large_image_array = np.zeros((num_images * image_size, total_width, 3), dtype=np.uint8)

# Loop through each image number
for image_number in range(num_images):
    # Loop through each model key
    offset = 0
    for model_idx, model_key in enumerate(model_keys):
        # Loop through each slice index
        for slice_idx, slice_index in enumerate(slice_indices):
            # Construct the image file name
            image_file_name = f'{model_key}_{image_number}_Axial{slice_index}.png'
            image_file_path = os.path.join(image_dir, image_file_name)

            # Check if the image file exists
            if os.path.exists(image_file_path):
                # Open the image file and convert it to RGB
                image = Image.open(image_file_path).convert('RGB')

                # Add the image to the large image array
                large_image_array[image_number * image_size:(image_number + 1) * image_size, (model_idx * len(slice_indices) + slice_idx) * image_size + offset:(model_idx * len(slice_indices) + slice_idx + 1) * image_size + offset] = np.array(image)
            else:
                print(f'Warning: Image file {image_file_name} does not exist.')
        
        # Increment the offset for the next model key
        offset += gap_size

# Save the large image array as a single image file
large_image = Image.fromarray(large_image_array)
large_image.save('large_image.png')