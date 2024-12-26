import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ...config import MRI2PETConfig
from ..models.paDiffusion import DiffusionModel

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

pretrain_dataset = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
pretrain_dataset = [(mri_path, os.path.join(config.mri_pretrain_dir, mri_path.split('/')[-1])) for mri_path in pretrain_dataset]
train_dataset = pickle.load(open('./src/data/trainDataset.pkl', 'rb'))
val_dataset = pickle.load(open('./src/data/valDataset.pkl', 'rb'))
LAMBDA1 = 0.5
LAMBDA2 = 0.5
LAMBDA3 = 0.05

if os.path.exists("./src/data/globalMeanPretrain.pkl"):
    global_mean = pickle.load(open("./src/data/globalMeanPretrain.pkl", "rb"))
    global_std = pickle.load(open("./src/data/globalStdPretrain.pkl", "rb"))
else:
    means = []
    variances = []
    for _, image in pretrain_dataset:
        img = np.load(image)
        means.append(np.mean(img))
        variances.append(np.var(img))

    global_mean = np.mean(means)
    global_variance = np.mean(variances) + np.mean((np.array(means) - global_mean) ** 2)
    global_std = np.sqrt(global_variance)
    pickle.dump(global_mean, open("./src/data/globalMeanPretrain.pkl", "wb"))
    pickle.dump(global_std, open("./src/data/globalStdPretrain.pkl", "wb"))

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if is_mri:
        img += torch.randn(img.size())
    else:
        img = (img - global_mean) / global_std
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

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

model = DiffusionModel(config).to(device)
if os.path.exists(f"./src/save/paDiffusion.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/paDiffusion.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])

steps_per_batch = 8
config.batch_size = config.batch_size // steps_per_batch
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(pretrain_dataset)
    pretrain_losses = []
    model.train()
    curr_step = 0
    optimizer.zero_grad()
    for i in range(0, len(pretrain_dataset), config.batch_size):
        batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)
        loss, _ = model(batch_context, batch_images, gen_loss=True)
        pretrain_losses.append(loss.cpu().detach().item())
        loss = loss / steps_per_batch
        loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            curr_step = 0

    cur_pretrain_loss = np.mean(pretrain_losses)
    print("Epoch %d Training Loss:%.7f"%(e, cur_pretrain_loss), flush=True)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mode': 'pretrain'
    }
    torch.save(state, f'./src/save/paDiffusion_base.pt')

model_S = deepcopy(model)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    train_losses = []
    model.train()
    curr_step = 0
    optimizer.zero_grad()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        train_losses.append(loss.cpu().detach().item())
        l_simple, x_T = model(batch_context, batch_images, gen_loss=True, output_images=True)
        _, x_S = model_S(batch_context, batch_images, gen_loss=True, output_images=True)
        l_img = model.compute_pairwise_sim_loss(x_S, x_T)
        l_hf, l_hfmse = model.compute_high_freq_loss(x_S, x_T, batch_images)
        loss = l_simple + LAMBDA1 * l_img + LAMBDA2 * l_hf + LAMBDA3 * l_hfmse
        loss = loss / steps_per_batch
        loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            curr_step = 0
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for v_i in range(0, len(val_dataset), config.batch_size):
            batch_context, batch_images = get_batch(val_dataset, v_i, config.batch_size)                 
            val_loss, _ = model(batch_context, batch_images, gen_loss=True)
            val_losses.append((val_loss).cpu().detach().item())
        
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss), flush=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mode': 'train'
        }
        torch.save(state, f'./src/save/paDiffusion.pt')
