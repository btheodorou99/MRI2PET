# GOALS:
# Synthetic performs similar to real PET in limited data settings
# Generated PET unlocks larger dataset to perform better than limited settings

import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from ..config import MRI2PETConfig
from ..models.downstreamModel import ImageClassifier


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
    subject = fname.split('-')[1]
    return subject

adni_labels = pd.read_csv('./src/data/DIAGNOSIS.csv')
adni_labels = {row.PTID: row.DIAGNOSIS for row in adni_labels.itertuples()}
adni_labels = {p: int(int(l - 1) == 2) for p, l in adni_labels.items() if l == l}
config.downstream_dim = 1

real_paired_dataset = [(m, p) for (m, p) in pickle.load(open('./src/data/trainDataset.pkl', 'rb')) + pickle.load(open('./src/data/valDataset.pkl', 'rb')) if getID(m) in adni_labels]
test_dataset = [(m, p) for (m, p) in pickle.load(open('./src/data/testDataset.pkl', 'rb')) if getID(m) in adni_labels]
train_mri = set([m for (m, p) in real_paired_dataset])
test_mri = set([m for (m, p) in test_dataset])

synthetic_dataset = [(m, p) for (m, p) in pickle.load(open('./src/data/syntheticDataset.pkl', 'rb')) if getID(m) in adni_labels and m not in test_mri]
synthetic_paired_dataset = [(m, p) for (m, p) in synthetic_dataset if m in train_mri]
augmented_paired_dataset = real_paired_dataset + [(m, p) for (m, p) in synthetic_dataset if m not in train_mri]

mri_dataset = [(m, None) for m in pickle.load(open('./src/data/mriDataset.pkl', 'rb')) if '/ADNI/' in m and getID(m) in adni_labels and m not in train_mri and m not in test_mri] + real_paired_dataset

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
    batch_labels = torch.zeros(bs, 1, dtype=torch.float)
    for i, (m, p) in enumerate(image_paths):
        batch_labels[i] = adni_labels[getID(m)]
        if has_mri:
            batch_mri[i] = load_image(m)
        if has_pet:
            batch_pet[i] = load_image(p)

    return batch_mri.to(device), batch_pet.to(device), batch_labels.to(device)

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

def train_model(train_data, has_mri, has_pet, key, seed):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    shuffle_training_data(train_data)
    train_data, val_data = train_data[:int(0.8*len(train_data))], train_data[int(0.8*len(train_data)):]
    model = ImageClassifier(config, has_mri, has_pet).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.downstream_lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    curr_patience = 0
    for e in tqdm(range(config.downstream_epoch), desc=f"Training {key} Model", leave=False):
        shuffle_training_data(train_data)
        model.train()
        for i in range(0, len(train_data), config.downstream_batch_size):
            mri, pet, labels = get_batch(train_data, i, config.batch_size, has_mri, has_pet)
            optimizer.zero_grad()
            output = model(mri, pet)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        running_loss = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_data), config.downstream_batch_size):
                mri, pet, labels = get_batch(val_data, i, config.batch_size, has_mri, has_pet)
                output = model(mri, pet)
                loss = criterion(output, labels)
                running_loss.append(loss.cpu().detach().item())
        val_loss = np.mean(running_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            curr_patience = 0
            torch.save(model.state_dict(), f'./src/save/downstream_binary_{key}.pt')
        else:
            curr_patience += 1

        if curr_patience >= config.downstream_patience:
            break

    model.load_state_dict(torch.load(f'./src/save/downstream_binary_{key}.pt'))
    return model

def evaluate_model(model, data, has_mri, has_pet):
    model.eval()
    prob_list = []
    label_list = []
    with torch.no_grad():
        for i in range(0, len(data), config.downstream_batch_size):
            mri, pet, labels = get_batch(data, i, config.batch_size, has_mri, has_pet)
            output = model(mri, pet)
            probs = torch.sigmoid(output)
            prob_list.extend(probs.tolist())
            label_list.extend(labels.tolist())

    probs = np.array(prob_list)
    labels = np.array(label_list)
    preds = np.round(probs)

    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, zero_division=0.0)
    recall = metrics.recall_score(labels, preds, zero_division=0.0)
    f1 = metrics.f1_score(labels, preds, zero_division=0.0)
    auroc = metrics.roc_auc_score(labels, probs)
    
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUROC': auroc,
    }
    return metrics_dict

experiments = [
    ('RealMRI', True, False, mri_dataset),
    ('RealLimitedMRI', True, False, real_paired_dataset),
    ('RealPET', False, True, real_paired_dataset),
    ('RealPaired', True, True, real_paired_dataset),
    ('SyntheticPET', False, True, synthetic_paired_dataset),
    ('AugmentedPET', False, True, augmented_paired_dataset),
    ('SyntheticPaired', True, True, synthetic_paired_dataset),
    ('AugmentedPaired', True, True, augmented_paired_dataset),
]

for key, hasMRI, hasPET, data in tqdm(experiments):
    print(key)
    
    metrics_dict = {}
    for run in tqdm(range(NUM_RUNS), desc="Training Runs", leave=False):
        seed = run * SEED
        model = train_model(data, hasMRI, hasPET, key, seed)
        run_dict = evaluate_model(model, test_dataset, hasMRI, hasPET)
        for k in run_dict:
            metrics_dict[k] = metrics_dict.get(k, []) + [run_dict[k]]

    metrics_dict = {k: (np.mean(v), np.std(v) / np.sqrt(NUM_RUNS)) for k, v in metrics_dict.items()}
    for k, v in metrics_dict.items():
        print(f"\t{k}: {v[0]:.5f} \\pm {v[1]:.5f}")

    pickle.dump(metrics_dict, open(f'./src/results/quantitative_evaluations/utility_binary_{key}.pkl', 'wb'))
