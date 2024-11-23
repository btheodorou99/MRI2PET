import os
import ants
import pickle
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split

pet_dir = "/data/CARD_AA/data/ADNI/PET_Nifti_PreProcessed/"
adni_dir = "/data/CARDPB/data/ADNI/MRI/"
ppmi_dir = "/data/CARDPB/data/PPMI/MRI/"
ukbb_dir = "/data/CARDPB/data/UKBB/MRI/"

def save_data(file_path, data): 
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def find_pet(pet_dir):
    data_dict = {}
    for niix_file in tqdm(os.listdir(pet_dir)):
        subject_id, date, _ = niix_file[:-4].split('--')  
        fpath = os.path.join(pet_dir, niix_file)
        date = datetime.strptime(date, "%Y-%m-%d")
        if subject_id not in data_dict:
            data_dict[subject_id] = {}
        data_dict[subject_id][date] = {'shape': ants.image_read(fpath).numpy().shape, 'filename': fpath.replace('PET_Nifti_PreProcessed', 'PET').replace('.nii', '.npy')}
    return data_dict

def find_mri(mri_dir, prefix, parseDate=False):
    data_dict = {}
    for subject_id in tqdm(os.listdir(mri_dir)):
        for date in os.listdir(os.path.join(mri_dir, subject_id)):
            for imaging_id in os.listdir(os.path.join(mri_dir, subject_id, date, 'T1wHierarchical')):
                fname = f'{prefix}-{subject_id}-{date}-T1wHierarchical-{imaging_id}-brain_n4_dnz.nii.gz'
                if not os.path.exists(os.path.join(mri_dir, subject_id, date, 'T1wHierarchical', imaging_id, fname)):
                    continue

                fpath = os.path.join(mri_dir, subject_id, date, 'T1wHierarchical', imaging_id, fname)
                if parseDate:
                    saveDate = datetime.strptime(date, "%Y%m%d")
                else:
                    saveDate = date
                if subject_id not in data_dict:
                    data_dict[subject_id] = {}
                data_dict[subject_id][saveDate] = {'shape': ants.image_read(fpath).numpy().shape, 'filename': fpath}
    return data_dict

def convert_mri_path(fpath):
    fpath = fpath.replace('CARDPB', 'CARD_AA').replace('nii.gz', 'npy')
    start, end = fpath.split('MRI')
    fname = end.split('/')[-1]
    return start + 'MRI/' + fname

def filter_mri_images(mri_dict):
    filtered_mri_paths = []
    converted_filtered_mri_paths = []
    height_counter = Counter()
    channel_counter = Counter()

    for _, dates in mri_dict.items():
        for _, info in dates.items():
            height_counter[info['shape'][0]] += 1
            channel_counter[info['shape'][2]] += 1
            if len(info['shape']) == 3 and info['shape'][2] > 1:
                filtered_mri_paths.append(info['filename'])
                converted_filtered_mri_paths.append(convert_mri_path(info['filename']))

    return filtered_mri_paths, converted_filtered_mri_paths, height_counter, channel_counter

def map_pet_to_mri(mri_dict, pet_dict):
    height_counter = Counter()
    channel_counter = Counter()

    for subject_id, dates in pet_dict.items():
        for date, info in dates.items():
            height_counter[info['shape'][0]] += 1
            channel_counter[info['shape'][2]] += 1

    paired_images = []
    mri_dates = {sub_id: {date: convert_mri_path(info['filename']) for date, info in dates.items() if len(info['shape']) == 3 and info['shape'][2] > 1}
                 for sub_id, dates in mri_dict.items()}
    mri_dates = {sub_id: dates for sub_id, dates in mri_dates.items() if dates}

    for sub_id, pet_dates in pet_dict.items():
        if sub_id not in mri_dates or not mri_dates[sub_id]:
            continue

        for pet_date, pet_info in pet_dates.items():
            if not mri_dates[sub_id]:
                continue
            
            closest_mri_date = min(mri_dates[sub_id], key=lambda d: abs(d - pet_date))
            dateDistance = abs(closest_mri_date - pet_date).days
            if dateDistance > 365:
                continue
            
            paired_images.append((mri_dates[sub_id][closest_mri_date], pet_info['filename']))
            mri_dates[sub_id].pop(closest_mri_date)

    return paired_images, height_counter, channel_counter

# Build the MRI data dictionaries
adni_mri_dict = find_mri(adni_dir, 'ADNI', parseDate=True)
ppmi_mri_dict = find_mri(ppmi_dir, 'PPMI')
ukbb_mri_dict = find_mri(ukbb_dir, 'UKBB')
mri_dict = {**adni_mri_dict, **ppmi_mri_dict, **ukbb_mri_dict}

# Load the PET data dictionary
pet_dict = find_pet(pet_dir)

# Map PET to MRI
pet_mri_pairs, pet_heights, pet_channels = map_pet_to_mri(adni_mri_dict, pet_dict)
print("PET Heights:", pet_heights)
print("PET Channels:", pet_channels)

# Filter MRI images, count dimensions, and save the filtered MRI paths
original_mri_paths, converted_mri_paths, mri_heights, mri_channels = filter_mri_images(mri_dict)
print("MRI Heights:", mri_heights)
print("MRI Channels:", mri_channels)
print("MRI Paths:", len(converted_mri_paths))
save_data('./src/data/mriDataset.pkl', converted_mri_paths)
save_data('./src/data/original_mriDataset.pkl', original_mri_paths)

# Save the PET-MRI pairs
print("PET-MRI Pairs:", len(pet_mri_pairs))
save_data('./src/data/pet_mri_pairs.pkl', pet_mri_pairs)

# Split the PET-MRI pairs into train, validation, and test sets
train_pairs, test_pairs = train_test_split(pet_mri_pairs, test_size=0.1, random_state=0)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=0)

pickle.dump(train_pairs, open('./src/data/trainDataset.pkl', 'wb'))
pickle.dump(val_pairs, open('./src/data/valDataset.pkl', 'wb'))
pickle.dump(test_pairs, open('./src/data/testDataset.pkl', 'wb'))