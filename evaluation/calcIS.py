import torch
import pickle
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=True, transform_input=False).to(device)
model.fc = torch.nn.Identity()
transform = Compose([Resize(299), CenterCrop(299), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_inception_score(model, dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            output = model(inputs)
            preds.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    kl_div = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    is_score = np.exp(np.mean(np.sum(kl_div, 1)))
    return is_score

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
    is_score = get_inception_score(model, model_dataloader)
    print('{k} Inception Score:', is_score)
    pickle.dump(is_score, open(f'../results/quantitative_evaluations/{k}_is.pkl', 'wb'))
