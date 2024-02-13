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
from pypet2bids.ecat import Ecat

pet_dir = "../data/ADNI_PET/"
output_dir = "../data/PET_Nifti/"

# Create a temporary directory
tempdir = "./temp/"
if not os.path.exists(tempdir):
    os.makedirs(tempdir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

missing = []

for subject_id in tqdm(os.listdir(pet_dir)):
    subject_path = os.path.join(pet_dir, subject_id)
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
                
                nii_filename = f"{subject_id}--{date}--{some_ids}.nii"
                output_filename = os.path.join(output_dir, nii_filename)
                if os.path.exists(output_filename):
                    continue

                file = files[0]
                try:
                    if file.lower().endswith('.dcm'):
                        # Convert DICOM to NIfTI
                        subprocess.run(["dcm2niix", "-o", tempdir, some_ids_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    elif file.lower().endswith('.v'):
                        # Convert ECAT to NIfTI
                        file_path = os.path.join(some_ids_path, file)
                        ecat = Ecat(ecat_file=file_path, nifti_file=os.path.join(tempdir, 'output.nii'), collect_pixel_data=True)
                        ecat.make_nifti()

                    # Process the NIfTI file
                    nifti_files = [f for f in os.listdir(tempdir) if f.endswith(('.nii', '.nii.gz'))]
                    if nifti_files:
                        niix_file = os.path.join(tempdir, nifti_files[0])
                        os.rename(niix_file, output_filename)
                except:
                    missing.append(nii_filename)

                for f in os.listdir(tempdir):
                    os.remove(os.path.join(tempdir, f))

os.rmdir(tempdir)
pickle.dump(missing, open('../data/missing_pet.pkl', 'wb'))