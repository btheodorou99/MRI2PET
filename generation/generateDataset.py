import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from ..config import MRI2PETConfig
from ..models.diffusionModel import DiffusionModel
from ..models.ganModel import Generator

SEED = 4
cudaNum = 0
NUM_SAMPLES = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

test_dataset = pickle.load(open('../data/testDataset.pkl', 'rb'))
test_dataset = [(mri_path, _) for (mri_path, _) in test_dataset]
image_transform = transforms.Compose([
        transforms.Resize((config.mri_image_dim, config.mri_image_dim)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 2*x - 1)  # Normalize to [-1, 1]
    ])

def load_image(image_path):
    with Image.open(f'{config.image_dir}/{image_path}.jpg') as img:
        return image_transform(img)

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_channels, config.mri_image_dim, config.mri_image_dim, device=device)
    for i, p in enumerate(image_paths):
        batch_context[i] = load_image(p)
        
    return batch_context
  
def tensor_to_image(tensor):
    # First de-normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert to PIL image
    img = transforms.ToPILImage()(tensor)
    return img
        
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
    print(k)
    os.makedirs(f'../results/generated_datasets/{k}', exist_ok=True)
    if 'GAN' in k:
        model = Generator(config).to(device)
        model.load_state_dict(torch.load(f'../save/{k}.pt', map_location=torch.device(device))['generator'])
        model.eval()
    else:
        model = DiffusionModel(config).to(device)
        model.load_state_dict(torch.load(f'../save/{k}.pt', map_location=torch.device(device))['model'])
        model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), config.batch_size), leave=False):
            sample_contexts = get_batch(test_dataset, i, config.batch_size)
            sample_images = model.generate(sample_contexts)
            for j in range(sample_images.size(0)):
                sample_image = tensor_to_image(sample_images[j].cpu())
                sample_image.save(f'../results/generated_datasets/{k}/sampleImage_{i+j}.jpg')