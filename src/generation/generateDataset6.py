import os
import torch
import random
import pickle
import importlib
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig
from ..models.ganModel import Generator
from ..models.diffusionModel import DiffusionModel

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

test_dataset = pickle.load(open('./src/data/testDataset.pkl', 'rb'))
test_dataset = [mri_path for (mri_path, pet_path) in test_dataset]

def load_image(image_path):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    return img

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, device=device)
    for i, p in enumerate(image_paths):
        batch_context[i] = load_image(p)
        
    return batch_context
  
def save_image(tensor, path):
    """Save a torch tensor as an image."""
    # Convert the tensor to a PIL image and save
    image = tensor.cpu().clone()
    image = (image + 1) / 2.0
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    np.save(path, image)
        
model_keys = [
    # 'baseGAN',
    # 'baseDiffusion',
    # 'proposedModel14',
    # 'roundTwo1',
    # 'roundTwo2',
    # 'roundTwo3',
    # 'roundTwo4',
    'roundTwo5',
    # 'roundTwo6',
    # 'roundTwo7',
    # 'noisyPretrainedDiffusion',
    # 'selfPretrainedDiffusion',
    # 'stylePretrainedDiffusion',
    # 'noisyPretrainedGAN',
    # 'selfPretrainedGAN',
    # 'stylePretrainedGAN',
    # 'mri2pet',
]

for k in tqdm(model_keys):
    print(k)
    os.makedirs(f'/data/theodoroubp/MRI2PET/results/generated_datasets/{k}', exist_ok=True)
    if 'GAN' in k:
        model = Generator(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['generator'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size
    elif 'proposed' in k:
        module_path = f"src.models.{k}"
        module = importlib.import_module(module_path)
        Model = getattr(module, 'DiffusionModel')
        if k == 'proposedModel3':
            pretrain_dataset = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
            mean_mri = torch.zeros(config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float, device=device)
            for i in tqdm(range(0, len(pretrain_dataset), config.batch_size)):
                batch_context = get_batch(pretrain_dataset, i, config.batch_size)
                mean_mri += torch.sum(batch_context, dim=0)
            mean_mri /= len(pretrain_dataset)
            model = Model(config, mean_mri)
        else:
            model = Model(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size // 5
    elif 'roundTwo' in k:
        module_path = f"src.models.{k}"
        module = importlib.import_module(module_path)
        Model = getattr(module, 'DiffusionModel')
        if k == 'roundTwo4':
            pretrain_dataset = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
            mean_mri = torch.zeros(config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float, device=device)
            for i in tqdm(range(0, len(pretrain_dataset), config.batch_size)):
                batch_context = get_batch(pretrain_dataset, i, config.batch_size)
                mean_mri += torch.sum(batch_context, dim=0)
            mean_mri /= len(pretrain_dataset)
            model = Model(config, mean_mri)
        else:
            model = Model(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size // 5
    else:
        model = DiffusionModel(config)
        model.load_state_dict(torch.load(f'./src/save/{k}.pt', map_location='cpu')['model'])
        model = model.to(device)
        model.eval()
        batch_size = config.batch_size // 5

    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), leave=False):
            sample_contexts = get_batch(test_dataset, i, batch_size)
            sample_images = model.generate(sample_contexts)
            for j in range(sample_images.size(0)):
                save_image(sample_images[j], f'/data/theodoroubp/MRI2PET/results/generated_datasets/{k}/sampleImage_{i+j}.npy')