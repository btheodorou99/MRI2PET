import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
from ..config import MRI2PETConfig
from ..models.downstreamModel import ImageRegressor

SEED = 4
NUM_RUNS = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def getID(path):
    fname = path.split('/')[-1]
    subject, date_str = fname.split('-')[1:3]
    # Convert date string to datetime object
    date = datetime.strptime(date_str, '%Y%m%d')
    return subject, date

# Load and process MMSE scores
adni_scores = pd.read_csv('./src/data/MMSE.csv')
# Convert VISDATE to datetime
adni_scores['VISDATE'] = pd.to_datetime(adni_scores['VISDATE'])
# Filter valid MMSE scores
adni_scores = adni_scores[(adni_scores['MMSCORE'] >= 1) & (adni_scores['MMSCORE'] <= 30)]

real_paired_dataset = pickle.load(open('./src/data/trainDataset.pkl', 'rb')) + pickle.load(open('./src/data/valDataset.pkl', 'rb'))
test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))
synthetic_dataset = pickle.load(open('./src/data/syntheticDataset.pkl', 'rb'))

# Create dictionary to store scores by subject
adni_scores_dict = {}
for _, row in adni_scores.iterrows():
    if row['PTID'] not in adni_scores_dict:
        adni_scores_dict[row['PTID']] = []
    adni_scores_dict[row['PTID']].append((row['VISDATE'], row['MMSCORE']))

def get_closest_score(subject, date):
    if subject not in adni_scores_dict:
        return None
    
    scores = adni_scores_dict[subject]
    # Find closest date
    closest_score = min(scores, key=lambda x: abs((x[0] - date).days))
    # Only use scores within 1 year of the scan
    if abs((closest_score[0] - date).days) > 365:
        return None
    return closest_score[1]

# Create labels dictionary
adni_labels = {}
for dataset in [real_paired_dataset, test_dataset, synthetic_dataset]:
    for m, _ in dataset:
        subject, date = getID(m)
        score = get_closest_score(subject, date)
        if score is not None:
            adni_labels[m] = score

# Filter datasets to only include images with valid MMSE scores
real_paired_dataset = [(m, p) for (m, p) in real_paired_dataset if m in adni_labels]
test_dataset = [(m, p) for (m, p) in test_dataset if m in adni_labels]
train_mri = set([m for (m, p) in real_paired_dataset])
test_mri = set([m for (m, p) in test_dataset])

# Update derived datasets
synthetic_dataset = [(m, p) for (m, p) in synthetic_dataset if m in adni_labels and m not in test_mri]
synthetic_paired_dataset = [(m, p) for (m, p) in synthetic_dataset if m in train_mri]
augmented_paired_dataset = real_paired_dataset + [(m, p) for (m, p) in synthetic_dataset if m not in train_mri]
mri_dataset = [(m, None) for m in pickle.load(open('./src/data/mriDataset.pkl', 'rb')) 
               if '/ADNI/' in m and m in adni_labels and m not in train_mri and m not in test_mri] + real_paired_dataset

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def get_batch(dataset, loc, batch_size, has_mri=False, has_pet=False):
    assert has_pet or has_mri, "At least one of MRI or PET must be present"
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_mri = torch.zeros(bs, config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float)
    batch_pet = torch.zeros(bs, config.n_pet_channels, config.pet_image_dim, config.pet_image_dim, dtype=torch.float)
    batch_labels = torch.zeros(bs, dtype=torch.float)
    for i, (m, p) in enumerate(image_paths):
        batch_labels[i] = adni_labels[m]
        if has_mri:
            batch_mri[i] = load_image(m)
        if has_pet:
            batch_pet[i] = load_image(p)
        
    return batch_mri.to(device), batch_pet.to(device), batch_labels.to(device)

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

def evaluate_model(model, data, has_mri, has_pet, return_predictions=False):
    model.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for i in range(0, len(data), config.downstream_batch_size):
            mri, pet, labels = get_batch(data, i, config.batch_size, has_mri, has_pet)
            output = model(mri, pet)
            pred_list.extend(output.cpu().numpy().flatten())
            label_list.extend(labels.cpu().numpy().flatten())

    preds = np.array(pred_list)
    labels = np.array(label_list)
    
    mse = metrics.mean_squared_error(labels, preds)
    mae = metrics.mean_absolute_error(labels, preds)
    r2 = metrics.r2_score(labels, preds)
    correlation = np.corrcoef(labels, preds)[0,1]
    
    metrics_dict = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Correlation': correlation,
    }
    
    if return_predictions:
        return metrics_dict, preds
    return metrics_dict

# Additional comparison between real and synthetic predictions
synthetic_test_dataset = {m: p for m, p in pickle.load(open('./src/data/syntheticDataset.pkl', 'rb')) if m in adni_labels and m in test_mri}
synthetic_test_dataset = [(m, synthetic_test_dataset[m]) for (m, _) in test_dataset]

# for key in [None, 'highLR', 'highScores', 'lowLR', 'noDropout', 'smallBatch']:
for key in [None]:
    print()
    print(key if key is not None else 'Default')

    model = ImageRegressor(config, False, True).to(device)
    model.load_state_dict(torch.load(f'./src/save/downstream_mmse{"" if key is None else "_" + key}_RealPET.pt'))
    _, real_preds_pet = evaluate_model(model, test_dataset, False, True, return_predictions=True)
    _, syn_preds_pet = evaluate_model(model, synthetic_test_dataset, False, True, return_predictions=True)

    model = ImageRegressor(config, True, True).to(device)
    model.load_state_dict(torch.load(f'./src/save/downstream_mmse{"" if key is None else "_" + key}_RealPaired.pt'))
    _, real_preds_paired = evaluate_model(model, test_dataset, True, True, return_predictions=True)
    _, syn_preds_paired = evaluate_model(model, synthetic_test_dataset, True, True, return_predictions=True)

    # Calculate Pearson correlations
    for normalization in ["TODO"]:
        normalized_real_preds_pet = real_preds_pet
        normalized_syn_preds_pet = syn_preds_pet
        normalized_real_preds_paired = real_preds_paired
        normalized_syn_preds_paired = syn_preds_paired

        pet_correlation = np.corrcoef(normalized_real_preds_pet, normalized_syn_preds_pet)[0,1]
        paired_correlation = np.corrcoef(normalized_real_preds_paired, normalized_syn_preds_paired)[0,1]

        print(f"PET Model Correlation: {pet_correlation:.4f}")
        print(f"Paired Model Correlation: {paired_correlation:.4f}")

        # Plot PET model results
        plt.figure(figsize=(6,5))
        plt.scatter(real_preds_pet, syn_preds_pet, alpha=0.5)
        plt.plot([0, 30], [0, 30], 'r--')  # Identity line
        plt.xlabel('Real PET Predictions')
        plt.ylabel('Synthetic PET Predictions')
        plt.title(f'PET Model (r={pet_correlation:.4f})')
        plt.tight_layout()
        plt.savefig(f'./src/results/quantitative_evaluations/mmse{"" if key is None else "_" + key}_{normalization}_pet_correlation.png')
        plt.close()

        # Plot paired model results 
        plt.figure(figsize=(6,5))
        plt.scatter(real_preds_paired, syn_preds_paired, alpha=0.5)
        plt.plot([0, 30], [0, 30], 'r--')  # Identity line
        plt.xlabel('Real Paired Predictions')
        plt.ylabel('Synthetic Paired Predictions')
        plt.title(f'Paired Model (r={paired_correlation:.4f})')
        plt.tight_layout()
        plt.savefig(f'./src/results/quantitative_evaluations/mmse{"" if key is None else "_" + key}_{normalization}_paired_correlation.png')
        plt.close()

        # Save results for plotting and future analysis
        comparison_results = {
            'real_preds_pet': real_preds_pet,
            'real_preds_paired': real_preds_paired, 
            'syn_preds_pet': syn_preds_pet,
            'syn_preds_paired': syn_preds_paired,
            'pet_correlation': pet_correlation,
            'paired_correlation': paired_correlation
        }
        pickle.dump(comparison_results, open(f'./src/results/quantitative_evaluations/mmse{"" if key is None else "_" + key}_{normalization}_real_vs_synthetic_comparison.pkl', 'wb'))
