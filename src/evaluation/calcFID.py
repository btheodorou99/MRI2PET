import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from scipy import linalg
from sklearn.utils import resample
from ..config import MRI2PETConfig
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=True).to(device)
model.fc = torch.nn.Identity()
transform = Compose([Resize(299), CenterCrop(299), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

BATCH_SIZE = 64 // config.n_pet_channels

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_image = np.zeros((bs, config.n_pet_channels, config.pet_image_dim, config.pet_image_dim))
    for i, p in enumerate(image_paths):
        batch_image[i] = load_image(p)
    
    batch_image = batch_image.reshape(bs * config.n_pet_channels, 1, config.pet_image_dim, config.pet_image_dim)
    batch_image = batch_image.repeat(3, axis=1)
    batch_image = transform(torch.from_numpy(batch_image)).float().to(device)
    return batch_image

def get_inception_features(model, dataset):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(dataset), BATCH_SIZE):
            batch_images = get_batch(dataset, i, BATCH_SIZE)
            batch_images = batch_images.to(device)
            output = model(batch_images)
            features.append(output.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return np.real(fid)


test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))
test_dataset = [pet_path for (mri_path, pet_path) in test_dataset]
test_act = get_inception_features(model, test_dataset)

model_keys = [
    # 'mri2pet_noPretrain',
    # 'baseDiffusion2',
    # 'mri2pet_noPretrain2',
    'mri2pet',
    'mri2pet_noLoss',
    'selfPretrainedDiffusion',
    'noisyPretrainedDiffusion'
]


for k in tqdm(model_keys):
    model_path = f'/data/theodoroubp/MRI2PET/results/generated_datasets/{k}/'
    model_dataset = [os.path.join(model_path, file) for file in os.listdir(model_path)]
    model_act = get_inception_features(model, model_dataset)
    fid_values = []
    for i in range(config.n_bootstrap):
        bs_indices = resample(np.arange(len(model_act)), replace=True)
        bs_real = test_act[bs_indices]
        bs_fake = model_act[bs_indices]
        fid_value = calculate_fid(bs_real, bs_fake)
        fid_values.append(fid_value)
    fid_value = (np.mean(fid_values), np.std(fid_values) / np.sqrt(config.n_bootstrap))
    print(f'{k} FID: {fid_value[0]:.3f} \\pm {fid_value[1]:.3f}')
    pickle.dump(fid_value, open(f'./src/results/quantitative_evaluations/{k}_fid.pkl', 'wb'))
