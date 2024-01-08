# ADNI PET SEARCH:
# Modality = PET
# Radiopharmaceutical = "18F-AV45"
# Frames?

# DIRECTORY STRUCTURE:
# ADNI/
#   subjectID/
#     seriesName/
#       date_otherStuff/
#         someIDs/
#           dicomFiles

import os
import pickle
import subprocess
import numpy as np
from tqdm import tqdm

import ants
from pypet2bids.ecat import Ecat

pet_dir = "./data/ADNI_PET/"
output_dir = "./data/PET/"
data_dict = {}

# Create a temporary directory
tempdir = "./temp/"
if not os.path.exists(tempdir):
    os.makedirs(tempdir)

for subject_id in tqdm(os.listdir(pet_dir)):
    subject_path = os.path.join(pet_dir, subject_id)
    if not os.path.isdir(subject_path):
        continue

    for series_name in os.listdir(subject_path):
        series_path = os.path.join(subject_path, series_name)
        for date_other in os.listdir(series_path):
            date = date_other.split('_')[0]
            date_path = os.path.join(series_path, date_other)
            for some_ids in os.listdir(date_path):
                some_ids_path = os.path.join(date_path, some_ids)
                files = os.listdir(some_ids_path)
                if not files:
                    continue

                file = files[0]
                if file.lower().endswith('.dcm'):
                    # Convert DICOM to NIfTI
                    subprocess.run(["dcm2niix", "-o", tempdir, file_path])
                elif file.lower().endswith('.v'):
                    # Convert ECAT to NIfTI
                    file_path = os.path.join(some_ids_path, file)
                    ecat = Ecat(ecat_file=file_path, nifti_file=os.path.join(tempdir, 'output.nii'), collect_pixel_data=True)
                    ecat.make_nifti()

                # Process the NIfTI file
                nifti_files = [f for f in os.listdir(tempdir) if f.endswith(('.nii', '.nii.gz'))]
                if nifti_files:
                    niix_file = os.path.join(tempdir, nifti_files[0])
                    img = ants.image_read(niix_file)
                    if len(img.shape) == 4:
                      num_time_points = img.shape[3]
                      reference_frame = img[:, :, :, 0]
                      registered_frames = [reference_frame]
                      for i in range(1, num_time_points):
                          moving_frame = img[:, :, :, i]
                          registration = ants.registration(fixed=ants.from_numpy(reference_frame),
                                                            moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
                          registered_frames.append(registration['warpedmovout'].numpy())

                      img = ants.from_numpy(np.mean(np.array(registered_frames), axis=0))
                      
                    data = img.numpy()
                    data = (data - data.min()) / (data.max() - data.min())

                    # Save processed data
                    npy_filename = f"{subject_id}_{date}.npy"
                    output_filename = os.path.join(output_dir, npy_filename)
                    np.save(output_filename, data)

                    # Update dictionary
                    if subject_id not in data_dict:
                        data_dict[subject_id] = {}
                    data_dict[subject_id][date] = {'shape': data.shape, 'filename': npy_filename}

                # Clean up temporary directory
                for f in os.listdir(tempdir):
                    os.remove(os.path.join(tempdir, f))
                
os.rmdir(tempdir)
pickle.dump(data_dict, open('./data/pet_dict.pkl', 'wb'))