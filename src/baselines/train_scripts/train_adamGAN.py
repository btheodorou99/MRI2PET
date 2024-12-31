import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from ...config import MRI2PETConfig
from ..models.adamGAN import Generator, Discriminator

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
FISHER_QUANTILE = 75

def load_image(image_path, is_mri=True):
    img = np.load(image_path)
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    if is_mri:
        img += torch.randn(img.size())
    else:
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

def shuffle_training_data(train_ehr_dataset):
    random.shuffle(train_ehr_dataset)

generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
# if os.path.exists(f"./src/save/adamGAN_base.pt"):
#     print("Loading previous model")
#     checkpoint = torch.load(f'./src/save/adamGAN_base.pt', map_location=torch.device(device))
#     generator.load_state_dict(checkpoint['generator'])
#     discriminator.load_state_dict(checkpoint['discriminator'])
#     optimizer_G.load_state_dict(checkpoint['optimizer_G'])
#     optimizer_D.load_state_dict(checkpoint['optimizer_D'])

# #########################
# ### PRETRAINING STAGE ###
# #########################

# for name, param in generator.named_parameters():
#     if '_vector' in name:
#         param.requires_grad = False

# for name, param in discriminator.named_parameters():
#     if '_vector' in name:
#         param.requires_grad = False

# for e in tqdm(range(config.pretrain_epoch)):
#     shuffle_training_data(pretrain_dataset)
#     generator.train()
#     discriminator.train()
#     for i in range(0, len(pretrain_dataset), config.batch_size):
#         batch_context, batch_images = get_batch(pretrain_dataset, i, config.batch_size)

#         # Train Discriminator
#         z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
#         fake_imgs = generator(z, batch_context)

#         real_validity = discriminator(batch_images, batch_context)
#         fake_validity = discriminator(fake_imgs, batch_context)
#         gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
#         d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty

#         optimizer_D.zero_grad()
#         d_loss.backward()
#         optimizer_D.step()

#         if i % (config.generator_interval * config.batch_size) == 0:
#             # Train Generator
#             fake_imgs = generator(z, batch_context)
#             fake_validity = discriminator(fake_imgs, batch_context)
#             g_loss = -torch.mean(fake_validity)

#             optimizer_G.zero_grad()
#             g_loss.backward()
#             optimizer_G.step()
#     state = {
#         'generator': generator.state_dict(),
#         'discriminator': discriminator.state_dict(),
#         'optimizer_G': optimizer_G,
#         'optimizer_D': optimizer_D,
#         'epoch': e
#     }
#     torch.save(state, f'./src/save/adamGAN_base.pt')


################################
### IMPORTANCE PROBING STAGE ###
################################

for name, param in generator.named_parameters():
    if '_vector' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

for name, param in discriminator.named_parameters():
    if '_vector' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# One Epoch of Fine Tuning
# print("Probing stage")
# probing_dataset = train_dataset[:-5*config.batch_size]
# for i in range(0, len(probing_dataset), config.batch_size):
#     batch_context, batch_images = get_batch(probing_dataset, i, config.batch_size)

#     # Train Discriminator
#     z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
#     fake_imgs = generator(z, batch_context)

#     real_validity = discriminator(batch_images, batch_context)
#     fake_validity = discriminator(fake_imgs, batch_context)
#     gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
#     d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty

#     optimizer_D.zero_grad()
#     d_loss.backward()
#     optimizer_D.step()

#     fake_imgs = generator(z, batch_context)
#     fake_validity = discriminator(fake_imgs, batch_context)
#     g_loss = -torch.mean(fake_validity)

#     optimizer_G.zero_grad()
#     g_loss.backward()
#     optimizer_G.step()

# state = {
#     'generator': generator.state_dict(),
#     'discriminator': discriminator.state_dict(),
#     'optimizer_G': optimizer_G,
#     'optimizer_D': optimizer_D,
# }
# torch.save(state, f'./src/save/adamGAN_probing.pt')
        
if os.path.exists(f"./src/save/adamGAN_probing.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'./src/save/adamGAN_probing.pt', map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])

for name, param in generator.named_parameters():
    param.requires_grad = True

for name, param in discriminator.named_parameters():
    param.requires_grad = True

probing_dataset = train_dataset[-5*config.batch_size:]
filter_fisher_g = dict()        
filter_fisher_d = dict()

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)

print("entering evaluation of fisher information...")
for i in range(0, len(probing_dataset), config.batch_size):
    batch_context, batch_images = get_batch(probing_dataset, i, config.batch_size)
    noise_fisher = torch.randn(batch_images.size(0), config.z_dim, device=batch_context.device)
    for fisher_idx in range(noise_fisher.size(0)):
        generator.zero_grad()
        discriminator.zero_grad()
        img_noise = noise_fisher[fisher_idx].view(1,-1)
        img_context = batch_context[fisher_idx].view(1,config.n_mri_channels,config.mri_image_dim,config.mri_image_dim)
        img_real = batch_images[fisher_idx].view(1,config.n_pet_channels,config.pet_image_dim,config.pet_image_dim)

        # 1) Obtain predicted results
        img_fake = generator(img_noise, img_context)
        fake_pred_fisher = discriminator(img_fake, img_context)
        real_pred_fisher = discriminator(img_real, img_context)
        g_loss_fisher = -torch.mean(fake_pred_fisher)
        d_loss_fisher = -torch.mean(real_pred_fisher) + torch.mean(fake_pred_fisher)

        # 2) Estimate the fisher information and grad of each parameter
        g_grads, est_fisher_info_g   = generator.estimate_fisher(loglikelihood=g_loss_fisher)
        d_grads, est_fisher_info_d   = discriminator.estimate_fisher(loglikelihood=d_loss_fisher)

        # FIM
        # Record filter-level FIM in G
        for key in est_fisher_info_g:
            if i == 0 and fisher_idx == 0:
                filter_fisher_g[key]  = est_fisher_info_g[key].detach().cpu().numpy()
            else:
                filter_fisher_g[key] += est_fisher_info_g[key].detach().cpu().numpy()
        
        # Record filter-level FIM in D
        for key in est_fisher_info_d:
            if i == 0 and fisher_idx == 0:
                filter_fisher_d[key]  = est_fisher_info_d[key].detach().cpu().numpy()
            else:
                filter_fisher_d[key] += est_fisher_info_d[key].detach().cpu().numpy()

# avg
# for key in filter_grad_g:
#     filter_grad_g[key]    /= (len(probing_dataset))
# for key in filter_grad_d:
#     filter_grad_d[key]    /= (len(probing_dataset))

for key in filter_fisher_g:
    filter_fisher_g[key]  /= (len(probing_dataset)) 
for key in filter_fisher_d:
    filter_fisher_d[key]  /= (len(probing_dataset)) 

# Obtain the quantile values for FC and Conv Layers
# G: FC
grouped_fim_fc_g = []
fc_names = ['init_map']
for name in fc_names:
    u_fim    = filter_fisher_g[f'{name}.weight'].mean()
    b_fim    = filter_fisher_g[f'{name}.bias']
    fim      = (u_fim + b_fim) / 2
    grouped_fim_fc_g = np.concatenate((grouped_fim_fc_g, fim), axis=None)
cutline_g_fc = np.percentile(grouped_fim_fc_g, q=FISHER_QUANTILE)

# G: CONV
grouped_fim_conv_g = []
conv_names = list(set(['.'.join(name.split('.')[:-1]) for name in filter_fisher_g.keys() if '.'.join(name).split('.')[:-1] not in fc_names]))
for name in conv_names:
    u_fim    = filter_fisher_g[f'{name}.u_vector'].mean() if f'{name}.u_vector' in filter_fisher_g else filter_fisher_g[f'{name}.weight'].mean()
    v_fim    = filter_fisher_g[f'{name}.v_vector'] if f'{name}.v_vector' in filter_fisher_g else filter_fisher_g[f'{name}.bias']
    fim      = u_fim + v_fim
    grouped_fim_conv_g = np.concatenate((grouped_fim_conv_g, fim), axis=None)
cutline_g_conv = np.percentile(grouped_fim_conv_g, q=FISHER_QUANTILE)

# Decisions
idx_kml_filter_fisher_g = dict()
idx_ft_filter_fisher_g = dict()
for key in filter_fisher_g:
    if any([name in key for name in conv_names]) and 'u_vector' in key: # for Conv layer with u_vector in name
        # resemble FIM
        u_fim    = filter_fisher_g[key].mean()
        v_fim    = filter_fisher_g[key.replace('u_vector', 'v_vector')]
        fim      = u_fim + v_fim
        # apply heuristics
        idx_kml_filter_fisher_g[key.replace('u_vector', 'v_vector')] = np.where(fim >   cutline_g_conv)[0]
        idx_ft_filter_fisher_g[key.replace('u_vector', 'v_vector')]  = np.where(fim <=  cutline_g_conv)[0]

    elif any([name in key for name in fc_names]) and 'u_vector' in key:  # for FC layer with u_vector in name
        # resemble FIM
        u_fim    = filter_fisher_g[key].mean()
        v_fim    = filter_fisher_g[key.replace('u_vector', 'v_vector')]
        b_fim    = filter_fisher_g[key.replace('u_vector', 'b_vector')]
        fim      = (u_fim + v_fim + b_fim) / 2
        # apply heuristics
        idx_kml_filter_fisher_g[key.replace('u_vector', 'v_vector')] = np.where(fim >   cutline_g_fc)[0]
        idx_ft_filter_fisher_g[key.replace('u_vector', 'v_vector')]  = np.where(fim <=  cutline_g_fc)[0]
        idx_kml_filter_fisher_g[key.replace('u_vector', 'b_vector')] = np.where(fim >   cutline_g_fc)[0]
        idx_ft_filter_fisher_g[key.replace('u_vector', 'b_vector')]  = np.where(fim <=  cutline_g_fc)[0]

# D: FC
grouped_fim_fc_d = []
fc_names = ['context_emb.fc1', 'context_emb.fc2', 'image_emb.fc1', 'image_emb.fc2', 'model.0', 'model.2', 'model.5', 'model.8', 'model.11', 'model.14']
for name in fc_names:
    u_fim    = filter_fisher_d[f'{name}.u_vector'].mean()
    v_fim    = filter_fisher_d[f'{name}.v_vector']
    b_fim    = filter_fisher_d[f'{name}.b_vector']
    fim      = (u_fim + v_fim + b_fim) / 2
    grouped_fim_fc_d = np.concatenate((grouped_fim_fc_d, fim), axis=None)
cutline_d_fc = np.percentile(grouped_fim_fc_d, q=FISHER_QUANTILE)

# D: CONV
grouped_fim_conv_d = []
conv_names = list(set(['.'.join(name.split('.')[:-1]) for name in filter_fisher_d.keys() if '.'.join(name).split('.')[:-1] not in fc_names]))
for name in conv_names:
    u_fim    = filter_fisher_d[f'{name}.u_vector'].mean()
    v_fim    = filter_fisher_d[f'{name}.v_vector']
    fim      = u_fim + v_fim
    grouped_fim_conv_d = np.concatenate((grouped_fim_conv_d, fim), axis=None)
cutline_d_conv = np.percentile(grouped_fim_conv_d, q=FISHER_QUANTILE)

# Decisions
idx_kml_filter_fisher_d = dict()
idx_ft_filter_fisher_d = dict()
for key in filter_fisher_d:
    if any([name in key for name in conv_names]) and 'u_vector' in key: # for Conv layer with u_vector in name
        # resemble FIM
        u_fim    = filter_fisher_d[key].mean()
        v_fim    = filter_fisher_d[key.replace('u_vector', 'v_vector')]
        fim      = u_fim + v_fim
        # apply heuristics
        idx_kml_filter_fisher_d[key.replace('u_vector', 'v_vector')] = np.where(fim >   cutline_d_conv)[0]
        idx_ft_filter_fisher_d[key.replace('u_vector', 'v_vector')]  = np.where(fim <=  cutline_d_conv)[0]

    elif any([name in key for name in fc_names]) and 'u_vector' in key:  # for FC layer with u_vector in name
        # resemble FIM
        u_fim    = filter_fisher_d[key].mean()
        v_fim    = filter_fisher_d[key.replace('u_vector', 'v_vector')]
        b_fim    = filter_fisher_d[key.replace('u_vector', 'b_vector')]
        fim      = (u_fim + v_fim + b_fim) / 2
        # apply heuristics
        idx_kml_filter_fisher_d[key.replace('u_vector', 'v_vector')] = np.where(fim >   cutline_d_fc)[0]
        idx_ft_filter_fisher_d[key.replace('u_vector', 'v_vector')]  = np.where(fim <=  cutline_d_fc)[0]
        idx_kml_filter_fisher_d[key.replace('u_vector', 'b_vector')] = np.where(fim >   cutline_d_fc)[0]
        idx_ft_filter_fisher_d[key.replace('u_vector', 'b_vector')]  = np.where(fim <=  cutline_d_fc)[0]


#############################
### MAIN ADAPTATION STAGE ###
#############################

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)

# filter-level KML: zero-out no-update KML weights (G)
for name, param in generator.named_parameters():
    param.requires_grad = True
    if name in idx_ft_filter_fisher_g.keys():
        with torch.no_grad():
            param[idx_ft_filter_fisher_g[name]] = 0 # zero-out kml value

# filter-level KML: zero-out no-update KML weights (D)
for name, param in discriminator.named_parameters():
    param.requires_grad = True
    if name in idx_ft_filter_fisher_d.keys():
        with torch.no_grad():
            param[idx_ft_filter_fisher_d[name]] = 0 # zero-out kml value

steps_per_batch = 4
config.batch_size = config.batch_size // steps_per_batch

for e in tqdm(range(config.epoch)):
    shuffle_training_data(train_dataset)
    generator.train()
    discriminator.train()
    curr_step = 0
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()
    for i in range(0, len(train_dataset), config.batch_size):
        batch_context, batch_images = get_batch(train_dataset, i, config.batch_size)
        
        # Train Discriminator
        z = torch.randn(batch_context.size(0), config.z_dim, device=batch_context.device)
        fake_imgs = generator(z, batch_context)

        real_validity = discriminator(batch_images, batch_context)
        fake_validity = discriminator(fake_imgs, batch_context)
        gradient_penalty = discriminator.compute_gradient_penalty(batch_images.data, fake_imgs.data, batch_context.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.lambda_gp * gradient_penalty
        d_loss = d_loss / steps_per_batch
        d_loss.backward()
        curr_step += 1
        if curr_step % steps_per_batch == 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            # ---------------------------------------------------------------------------------------
            # D: zero-out grad for KML filters with *low* KML FIM
            for name, param in discriminator.named_parameters():
                if name in idx_ft_filter_fisher_d.keys():
                    param.grad[idx_ft_filter_fisher_d[name]] = 0

            # D: zero-out grad for pretrained filters with *high* KML FIM
            filter_name = []
            for name, _ in discriminator.named_parameters():
                if name in idx_kml_filter_fisher_d.keys():    
                    if 'v_vector' in name:        
                        filter_name.append(name.replace('v_vector', 'W'))
                    elif 'b_vector' in name:
                        filter_name.append(name.replace('b_vector', 'bias'))

            for name, param in discriminator.named_parameters():
                if name in filter_name:
                    if 'weight' in name:
                        corresponding_kml_name = name.replace('W', 'v_vector')
                        param.grad[idx_kml_filter_fisher_d[corresponding_kml_name]] = 0
                    elif 'bias' in name:
                        corresponding_kml_name = name.replace('bias', 'b_vector')
                        param.grad[idx_kml_filter_fisher_d[corresponding_kml_name]] = 0   
            # ---------------------------------------------------------------------------------------
            optimizer_D.step()
            optimizer_D.zero_grad()

        if i % (config.generator_interval * config.batch_size) == 0:
            # Train Generator
            fake_imgs = generator(z, batch_context)
            fake_validity = discriminator(fake_imgs, batch_context)
            g_loss = -torch.mean(fake_validity)
            g_loss = g_loss / steps_per_batch
            g_loss.backward()
            if (curr_step + 1) % (steps_per_batch * config.generator_interval) == 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
                # ---------------------------------------------------------------------------------------
                # G: zero-out grad for KML filters with *low* KML FIM
                for name, param in generator.named_parameters():
                    if name in idx_ft_filter_fisher_g.keys():
                        param.grad[idx_ft_filter_fisher_g[name]] = 0

                # G: zero-out grad for pretrained filters with *high* KML FIM
                filter_name = []
                for name, _ in generator.named_parameters():
                    if name in idx_kml_filter_fisher_g.keys():    
                        if 'v_vector' in name:        
                            filter_name.append(name.replace('v_vector', 'W'))
                        elif 'b_vector' in name:
                            filter_name.append(name.replace('b_vector', 'bias'))

                for name, param in generator.named_parameters():
                    if name in filter_name:
                        if 'weight' in name:
                            corresponding_kml_name = name.replace('W', 'v_vector')
                            param.grad[idx_kml_filter_fisher_g[corresponding_kml_name]] = 0
                        elif 'bias' in name:
                            corresponding_kml_name = name.replace('bias', 'b_vector')
                            param.grad[idx_kml_filter_fisher_g[corresponding_kml_name]] = 0   
                # ---------------------------------------------------------------------------------------
                optimizer_G.step()
                optimizer_G.zero_grad()
                curr_step = 0

    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'epoch': e
    }
    torch.save(state, f'./src/save/adamGAN.pt')
