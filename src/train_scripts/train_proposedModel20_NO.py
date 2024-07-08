# DO NOT USE

import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig
from ..models.proposedModel20 import DiffusionModel, ImagePairClassifier

SEED = 4
cudaNum = 0
LABEL_DIR = './src/data/patient_labels'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = MRI2PETConfig()
device = torch.device(f"cuda:{cudaNum}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_patient_labels(path):
    filename = path.split('/')[-1].split('.')[0]
    pat_id = filename.split('_')[0]
    label_path = os.path.join(LABEL_DIR, f'{pat_id}.txt')
    labels = np.array([float(l) for l in open(label_path, 'r').read().split('\n') if l != ''])
    return labels

pretrain_dataset = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
pretrain_dataset = [(mri_path, os.path.join(config.mri_style_dir, mri_path.split('/')[-1])) for mri_path in pretrain_dataset]
train_dataset = pickle.load(open('./src/data/trainDataset.pkl', 'rb'))
train_cls_dataset = [(mri_path, get_patient_labels(mri_path)) for (mri_path, _) in train_dataset] + [(pet_path, get_patient_labels(pet_path)) for (_, pet_path) in train_dataset]
val_dataset = pickle.load(open('./src/data/valDataset.pkl', 'rb'))
val_cls_dataset = [(mri_path, get_patient_labels(mri_path)) for (mri_path, _) in val_dataset] + [(pet_path, get_patient_labels(pet_path)) for (_, pet_path) in val_dataset]

NUM_LABELS = len(train_cls_dataset[0][1])
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

def get_batch_cls(dataset, loc, batch_size):
    image_paths = dataset[loc:loc+batch_size]
    bs = len(image_paths)
    batch_image = torch.zeros(bs, config.n_pet_channels, config.pet_image_dim, config.pet_image_dim, dtype=torch.float, device=device)
    batch_labels = torch.zeros(bs, NUM_LABELS, dtype=torch.float, device=device)
    for i, (img, l) in enumerate(image_paths):
        batch_context[i] = load_image(img)
        batch_labels[i] = torch.tensor(l)
        
    return batch_image, batch_labels

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

model = DiffusionModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(f"./src/save/proposedModel20.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/proposedModel20.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

steps_per_batch = 5
config.batch_size = config.batch_size // steps_per_batch

for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(pretrain_dataset)
    pretrain_losses = []
    model.train()
    curr_step = 0
    optimizer.zero_grad()
    for i in range(0, len(pretrain_dataset), config.batch_size):
        batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)
        optimizer.zero_grad()
        loss, _ = model(batch_context, batch_images, gen_loss=True)
        pretrain_losses.append(loss.cpu().detach().item())
        loss = loss / steps_per_batch
        loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
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
    torch.save(state, f'./src/save/proposedModel20.pt')
    
cls_model = ImagePairClassifier(config).to(device)
optimizer = torch.optim.Adam(cls_model.parameters(), lr=config.lr)
bce = torch.nn.BCELoss()
global_loss = 1e10
for e in tqdm(range(config.pretrain_epoch)):
    shuffle_training_data(train_cls_dataset)
    cls_losses = []
    cls_model.train()
    for i in range(0, len(train_cls_dataset), config.batch_size):
        batch_mri, batch_pet, batch_labels = get_batch_cls(train_cls_dataset, i, config.batch_size)
        preds = cls_model(batch_mri, batch_pet)
        loss = bce(preds, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cls_losses.append(loss.cpu().detach().item())
    
    cur_cls_loss = np.mean(pretrain_losses)
    cls_val_losses = []
    for i in range(0, len(val_cls_dataset), config.batch_size):
        batch_mri, batch_pet, batch_labels = get_batch_cls(val_cls_dataset, i, config.batch_size)
        preds = cls_model(batch_mri, batch_pet)
        loss = bce(preds, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cls_val_losses.append(loss.cpu().detach().item())
        
    cur_cls_val_loss = np.mean(cls_val_losses)
    print("Classifier Epoch %d Training Loss:%.7f Validation Loss:%.7f"%(e, cur_cls_loss, cur_cls_val_loss))
    if cur_cls_val_loss < global_loss:
        global_loss = cur_cls_val_loss
        state = {
            'model': cls_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, f'./src/save/proposedModel20_cls.pt')
        
cls_model.load_state_dict(torch.load(f'./src/save/proposedModel20_cls.pt')['model'])
cls_model.eval()
cls_model.requires_grad_(False)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    train_losses = []
    model.train()
    curr_step = 0
    optimizer.zero_grad()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        loss, _ = model(batch_context, batch_images, gen_loss=True, pairModel=cls_model)
        train_losses.append(loss.cpu().detach().item())
        loss = loss / steps_per_batch
        loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
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
        
        cur_train_loss = np.mean(train_losses)
        cur_val_loss = np.mean(val_losses)
        print("Epoch %d Training Loss: %.7f, Validation Loss:%.7f"%(e, cur_train_loss, cur_val_loss), flush=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mode': 'train'
        }
        torch.save(state, f'./src/save/proposedModel20.pt')