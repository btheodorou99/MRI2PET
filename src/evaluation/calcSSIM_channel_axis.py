import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from ..config import MRI2PETConfig
from skimage.metrics import structural_similarity as ssim

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def calculate_ssim(img1, img2):
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))
test_dataset = [pet_path for (mri_path, pet_path) in test_dataset]

model_keys = [
    'baseDiffusion',
    'baseGAN',
    'mri2pet',
    'mri2pet_pScale',
    'mri2pet_sameOptimizer',
    'mri2pet_smallerLoss',
    'mri2pet_noPretrain',
    'mri2pet_noLoss',
    'mri2pet_noLoss_pScale',
    'selfPretrainedDiffusion',
    'selfPretrainedDiffusion_pScale',
    'noisyPretrainedDiffusion',
    'noisyPretrainedDiffusion_pScale',
    'baseDiffusion_tweaked',
    'baseGAN_tweaked',
    'mri2pet_pScale_tweaked',
    'mri2pet_noPretrain_tweaked',
    'mri2pet_noLoss_pScale_tweaked',
    'selfPretrainedDiffusion_pScale_tweaked',
    # 'noisyPretrainedDiffusion_pScale_tweaked',
]

for k in tqdm(model_keys):
    model_path = f'/data/theodoroubp/MRI2PET/results/generated_datasets/{k}/'
    model_dataset = [os.path.join(model_path, file) for file in os.listdir(model_path)]
    model_dataset = sorted(model_dataset, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ssim_values = [calculate_ssim(load_image(test_dataset[i]), load_image(model_dataset[i])) for i in range(len(test_dataset))]
    for i in range(config.n_bootstrap):
        bs_indices = resample(np.arange(len(ssim_values)), replace=True)
        bs_scores = [ssim_values[i] for i in bs_indices]
        ssim_values.append(np.mean(bs_scores))
    ssim_value = (np.mean(ssim_values), np.std(ssim_values) / np.sqrt(config.n_bootstrap))
    print(f'{k} SSIM: {ssim_value[0]:.3f} \\pm {ssim_value[1]:.3f}')
    pickle.dump(ssim_value, open(f'./src/results/quantitative_evaluations/{k}_ssim_channel_axis.pkl', 'wb'))
