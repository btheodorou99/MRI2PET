import os
import pickle
import random
import subprocess
from tqdm import tqdm
from ..config import MRI2PETConfig

config = MRI2PETConfig()
pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
output_dir = "/data/CARD_AA/data/ADNI/PET_Nifti_PreProcessed/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# {subject_id}--{date}--{some_ids}.nii
# single_files = []
problem_files = []
allFiles = os.listdir(pet_dir)
allFiles = [f for f in allFiles if f.endswith('.nii') and not os.path.exists(os.path.join(output_dir, f))]
random.shuffle(allFiles)
for niix_file in tqdm(allFiles):
    if os.path.exists(os.path.join(output_dir, niix_file)):
        continue

    try:
        result = subprocess.run(
            ["python", "-m", "src.utils.PET_PreProcessingSingle", pet_dir, niix_file, output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f'Success: {niix_file}')
        else:
            problem_files.append(niix_file)
    except subprocess.CalledProcessError as e:
        problem_files.append(niix_file)
    except Exception as e:
        problem_files.append(niix_file)

# pickle.dump(single_files, open('./src/data/single_nii.pkl', 'wb'))
pickle.dump(problem_files, open('./src/data/problem_nii.pkl', 'wb'))