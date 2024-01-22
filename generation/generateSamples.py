import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
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

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if not is_mri:
        img = 2 * img - 1
    return img

def get_batch(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_context = torch.zeros(bs, config.n_mri_channels, config.mri_image_dim, config.mri_image_dim, dtype=torch.float, device=device)
    batch_image = torch.zeros(bs, config.n_pet_channels, config.pet_image_dim, config.pet_image_dim, dtype=torch.float, device=device)
    for i, (m, p) in enumerate(image_paths):
        batch_context[i] = load_image(m, is_mri=True)
        batch_image[i] = load_image(p, is_mri=False)
        
    return batch_context, batch_image
  
def tensor_to_image(tensor):
    # First de-normalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    # Convert to PIL image
    img = transforms.ToPILImage()(tensor)
    return img

# Generate Samples
sampleIdx = np.random.choice(len(test_dataset), size=NUM_SAMPLES)
sample_data = [test_dataset[i] for i in sampleIdx]
sample_contexts, real_images = get_batch(sample_data, 0, NUM_SAMPLES)
resized_context_images = [Image.open(mri_path).resize((config.pet_image_dim, config.pet_image_dim)) for mri_path, _ in test_dataset]

for i in tqdm(range(NUM_SAMPLES)):
    real_image = tensor_to_image(real_images[i].cpu())
    real_image.save(f'../results/image_samples/realImage_{i}.jpg')

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
    if 'GAN' in k:
        model = Generator(config).to(device)
        model.load_state_dict(torch.load(f'../save/{k}.pt', map_location=torch.device(device))['generator'])
        model.eval()
    else:
        model = DiffusionModel(config).to(device)
        model.load_state_dict(torch.load(f'../save/{k}.pt', map_location=torch.device(device))['model'])
        model.eval()

    with torch.no_grad():
        sample_images = model.generate(sample_contexts)

    num_per_row = NUM_SAMPLES ** 0.5
    grid = Image.new('RGB' if config.n_pet_channels == 3 else 'L', (config.pet_image_dim * num_per_row, config.pet_image_dim * num_per_row), color='white')
    paired_grid = Image.new('RGB' if config.n_pet_channels == 3 else 'L', (config.pet_image_dim * num_per_row * 2, config.pet_image_dim * num_per_row), color='white')
    for i in tqdm(range(NUM_SAMPLES)):
        sample_image = tensor_to_image(sample_images[i].cpu())
        sample_image.save(f'../results/image_samples/sampleImage_{i}.jpg')

        x = (i % num_per_row) * config.pet_image_dim
        y = (i // num_per_row) * config.pet_image_dim
        grid.paste(sample_images[i], (x, y))

        row = i // 5
        col = (i % 5) * 2

        x_main = col * config.pet_image_dim
        y_main = row * config.pet_image_dim
        paired_grid.paste(sample_images[i], (x_main, y_main))

        x_context = x_main + config.pet_image_dim
        y_context = y_main
        paired_grid.paste(resized_context_images[i], (x_context, y_context))

    
    grid.save(f'../results/image_grids/{k}.jpg')
    paired_grid.save(f'../results/image_grids/{k}_paired.jpg')