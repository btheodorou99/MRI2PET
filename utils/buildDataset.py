import pickle
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data_dict(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def filter_mri_images(mri_dict):
    filtered_mri_paths = []
    height_counter = Counter()
    channel_counter = Counter()

    for subject_id, dates in mri_dict.items():
        for date, info in dates.items():
            height_counter[info['shape'][0]] += 1
            channel_counter[info['shape'][2]] += 1
            if len(info['shape']) == 3 and info['shape'][2] > 1:  # Assuming the third dimension is channels
                filtered_mri_paths.append(info['filename'])

    return filtered_mri_paths, height_counter, channel_counter

def map_pet_to_mri(mri_dict, pet_dict):
    height_counter = Counter()
    channel_counter = Counter()

    for subject_id, dates in pet_dict.items():
        for date, info in dates.items():
            height_counter[info['shape'][0]] += 1
            channel_counter[info['shape'][2]] += 1
  
    paired_images = []
    mri_dates = {sub_id: {datetime.strptime(date, "%Y-%m-%d"): info['filename'] for date, info in dates.items() if info['shape'][2] > 1}
                 for sub_id, dates in mri_dict.items()}
    mri_dates = {sub_id: dates for sub_id, dates in mri_dates.items() if dates}

    for sub_id, pet_dates in pet_dict.items():
        if sub_id not in mri_dates or not mri_dates[sub_id]:
            continue

        for pet_date, pet_info in pet_dates.items():
            if not mri_dates[sub_id]:
                continue
            
            pet_date = datetime.strptime(pet_date, "%Y-%m-%d")
            closest_mri_date = min(mri_dates[sub_id], key=lambda d: abs(d - pet_date))
            dateDistance = abs(closest_mri_date - pet_date).days
            if dateDistance > 365:
                continue
            
            paired_images.append((mri_dates[sub_id][closest_mri_date], pet_info['filename']))
            mri_dates[sub_id].pop(closest_mri_date)

    return paired_images, height_counter, channel_counter

# Load the MRI data dictionary
mri_dict = load_data_dict('../data/mri_dict.pkl')

# Filter MRI images and count dimensions
filtered_mri_paths, mri_heights, mri_channels = filter_mri_images(mri_dict)
print("MRI Heights:", mri_heights)
print("MRI Channels:", mri_channels)

# Save the filtered MRI paths
print("MRI Paths:", len(filtered_mri_paths))
save_data('../data/mriDataset.pkl', filtered_mri_paths)

# Load the PET data dictionary
pet_dict = load_data_dict('../data/pet_dict.pkl')

# Map PET to MRI
pet_mri_pairs, pet_heights, pet_channels = map_pet_to_mri(mri_dict, pet_dict)
print("PET Heights:", pet_heights)
print("PET Channels:", pet_channels)

# Save the PET-MRI pairs
print("PET-MRI Pairs:", len(pet_mri_pairs))
save_data('../data/pet_mri_pairs.pkl', pet_mri_pairs)

# Split the PET-MRI pairs into train, validation, and test sets
train_pairs, test_pairs = train_test_split(pet_mri_pairs, test_size=0.1, random_state=0)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=0)

pickle.dump(train_pairs, open('../data/trainDataset.pkl', 'wb'))
pickle.dump(val_pairs, open('../data/valDataset.pkl', 'wb'))
pickle.dump(test_pairs, open('../data/testDataset.pkl', 'wb'))