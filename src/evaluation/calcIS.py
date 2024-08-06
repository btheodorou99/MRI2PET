import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from ..config import MRI2PETConfig
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=True, transform_input=False).to(device)
transform = Compose([Resize(299), CenterCrop(299), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

BATCH_SIZE = 64 // config.n_pet_channels

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
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

def get_inception_score(model, dataset):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(dataset), BATCH_SIZE):
            batch_images = get_batch(dataset, i, BATCH_SIZE)
            batch_images = batch_images.to(device)
            output = model(batch_images)
            preds.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    py = np.mean(preds, axis=0)
    scores = []    
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))
    
    return np.exp(np.mean(scores))

model_keys = [
    'baseGAN',
    'baseDiffusion',
    'noisyPretrainedDiffusion',
    'selfPretrainedDiffusion',
    'mri2pet',
    'mri2pet_noLoss',
    'mri2pet_noNoise',
    'mri2pet_noPretrain',
]

for k in tqdm(model_keys):
    model_path = f'/data/theodoroubp/MRI2PET/results/generated_datasets/{k}/'
    model_dataset = [os.path.join(model_path, file) for file in os.listdir(model_path)]
    is_score = get_inception_score(model, model_dataset)
    print(f'{k} Inception Score:', is_score)
    pickle.dump(is_score, open(f'./src/results/quantitative_evaluations/{k}_is.pkl', 'wb'))
