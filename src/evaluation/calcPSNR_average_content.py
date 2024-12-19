import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from ..config import MRI2PETConfig
from skimage.metrics import peak_signal_noise_ratio as psnr

config = MRI2PETConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def gen_content_idx(img, threshold=1.05):
    content_val = max(img[0].max(), img[-1].max()) * threshold
    return [
        slice_idx
        for slice_idx in range(config.n_pet_channels)
        if img[slice_idx].max() > content_val
    ]

def calculate_psnr(img1, img2):
    content_idx = gen_content_idx(img1)
    img1 = img1.cpu().numpy()[content_idx].transpose(1, 2, 0)
    img2 = img2.cpu().numpy()[content_idx].transpose(1, 2, 0)
    return np.mean(
        [psnr(img1[i], img2[i], data_range=1.0) for i in range(config.n_pet_channels)]
    )

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
    psnr_values = [calculate_psnr(load_image(test_dataset[i]), load_image(model_dataset[i])) for i in range(len(test_dataset))]
    for i in range(config.n_bootstrap):
        bs_indices = resample(np.arange(len(psnr_values)), replace=True)
        bs_scores = [psnr_values[i] for i in bs_indices]
        psnr_values.append(np.mean(bs_scores))
    psnr_value = (np.mean(psnr_values), np.std(psnr_values) / np.sqrt(config.n_bootstrap))
    print(f'{k} psnr: {psnr_value[0]:.3f} \\pm {psnr_value[1]:.3f}')
    pickle.dump(psnr_value, open(f'./src/results/quantitative_evaluations/{k}_psnr_average_content.pkl', 'wb'))
