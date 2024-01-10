import os
import shutil
from tqdm import tqdm

source_dir = '../data/ADNI_MRI3'
target_dir = '../data/ADNI_MRI'

# Iterate through each SubjectID in the source directory
for subject_id in tqdm(os.listdir(source_dir)):
    if subject_id == '.DS_Store':
        continue
    
    subject_path_source = os.path.join(source_dir, subject_id)
    subject_path_target = os.path.join(target_dir, subject_id)

    # Check if SubjectID exists in the target directory
    if not os.path.exists(subject_path_target):
        # If SubjectID does not exist in target, copy the entire directory
        shutil.copytree(subject_path_source, subject_path_target)
    else:
        # If SubjectID exists, check each Keyword
        for keyword in os.listdir(subject_path_source):
            if keyword == '.DS_Store':
                continue
            
            keyword_path_source = os.path.join(subject_path_source, keyword)
            keyword_path_target = os.path.join(subject_path_target, keyword)

            # Check if Keyword exists under the same SubjectID in target
            if not os.path.exists(keyword_path_target):
                # If Keyword does not exist in target, copy the entire directory
                shutil.copytree(keyword_path_source, keyword_path_target)
            else:
                # If Keyword exists, check each Date
                for date in os.listdir(keyword_path_source):
                    if date == '.DS_Store':
                        continue
                    
                    date_path_source = os.path.join(keyword_path_source, date)
                    date_path_target = os.path.join(keyword_path_target, date)

                    # Check if Date exists under the same Keyword in target
                    if not os.path.exists(date_path_target):
                        # If Date does not exist in target, copy the entire directory
                        shutil.copytree(date_path_source, date_path_target)
                    # If Date exists, you can add additional logic here as needed