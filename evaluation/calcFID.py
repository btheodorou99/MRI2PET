import torch
import pickle
import numpy as np
from tqdm import tqdm
from scipy import linalg
from ..config import MRI2PETConfig
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=True).to(device)
model.fc = torch.nn.Identity()
transform = Compose([Resize(299), CenterCrop(299), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_inception_features(model, dataloader):
    model.eval()
    features = []

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            output = model(inputs)
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


test_path = '../data/test_images/'
test_dataset = ImageFolder(root=test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_act = get_inception_features(model, test_dataloader)

model_keys = [
    'baseDiffusion',
    'baseGAN',
    'noisyPretrainedDiffusion',
    'noisyPretrainedGAN',
    'selfPretrainedDiffusion',
    'selfPretrainedGAN',
    'stylePretrainedDiffusion',
    'stylePretrainedGAN',
    'mri2pet',
]

for k in tqdm(model_keys):
    model_path = f'../results/generated_datasets/{k}/'
    model_dataset = ImageFolder(root=model_path, transform=transform)
    model_dataloader = DataLoader(model_dataset, batch_size=32, shuffle=False)
    model_act = get_inception_features(model, model_dataloader)
    fid_value = calculate_fid(test_act, model_act)
    print('{k} FID:', fid_value)
    pickle.dump(fid_value, open(f'../results/quantitative_evaluations/{k}_fid.pkl', 'wb'))
