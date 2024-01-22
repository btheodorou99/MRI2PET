# ADNI MRI SEARCH:
# Modality = MRI
# Weighting = "T1"
# Acquisition Plane = "Axial"
# Slice Thickness?

# DIRECTORY STRUCTURE:
# ADNI/
#   subjectID/
#     seriesName/
#       date_otherStuff/
#         someIDs/
#           dicomFiles

# MRI PROCESSING:
# dcm2niix -o tempdir/ ./dicomdir/
# import ants
# niix_file = "tempdir/" + os.listdir("tempdir/")[0][:-4] + ".nii"
# img = ants.image_read( "niix_file" )
# Registration? Maybe.
# Segmentation? Probably Not.
# img = ants.n4_bias_field_correction(img)
# data = img.numpy()
# data = ( data - data.min() )/( data.max()-data.min() )
# Save to NPY file and add metadata map to dataset
# img = ants.from_numpy( data )

import os
import pickle
import subprocess
import numpy as np
from tqdm import tqdm

import ants

mri_dir = "../data/ADNI_MRI/"
output_dir = "../data/MRI/"
data_dict = {}

# Create a temporary directory
tempdir = "./temp/"
if not os.path.exists(tempdir):
    os.makedirs(tempdir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subject_id in tqdm(os.listdir(mri_dir)):
    subject_path = os.path.join(mri_dir, subject_id)
    if not os.path.isdir(subject_path):
        continue

    for series_name in os.listdir(subject_path):
        series_path = os.path.join(subject_path, series_name)
        if not os.path.isdir(series_path):
            continue
        
        for date_other in os.listdir(series_path):
            date = date_other.split('_')[0]
            date_path = os.path.join(series_path, date_other)
            if not os.path.isdir(date_path):
                continue
            
            for some_ids in os.listdir(date_path):
                some_ids_path = os.path.join(date_path, some_ids)
                if not os.path.isdir(some_ids_path):
                    continue
                
                files = os.listdir(some_ids_path)
                if not files:
                    continue
                
                npy_filename = f"{subject_id}---{date}.npy"
                output_filename = os.path.join(output_dir, npy_filename)
                if os.path.exists(output_filename):
                    if subject_id not in data_dict:
                        data_dict[subject_id] = {}
                    data_dict[subject_id][date] = {'shape': np.load(output_filename).shape, 'filename': npy_filename}
                    continue

                # Convert DICOM to NIfTI
                subprocess.run(["dcm2niix", "-o", tempdir, some_ids_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Process the NIfTI file
                nifti_files = [f for f in os.listdir(tempdir) if f.endswith(('.nii', '.nii.gz')) and not f.endswith('.json')]
                if nifti_files:
                    niix_file = os.path.join(tempdir, nifti_files[0])
                    img = ants.image_read(niix_file)

                    # Bias field correction
                    img = ants.n4_bias_field_correction(img)

                    data = img.numpy()
                    data = (data - data.min()) / (data.max() - data.min())

                    # Save processed data
                    np.save(output_filename, data)

                    # Update dictionary
                    if subject_id not in data_dict:
                        data_dict[subject_id] = {}
                    data_dict[subject_id][date] = {'shape': data.shape, 'filename': npy_filename}

                # Clean up temporary directory
                for f in os.listdir(tempdir):
                    os.remove(os.path.join(tempdir, f))
                
os.rmdir(tempdir)
pickle.dump(data_dict, open('../data/mri_dict.pkl', 'wb'))