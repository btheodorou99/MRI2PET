import os
import ants
import pickle
import random
import numpy as np
from tqdm import tqdm
from ..config import MRI2PETConfig

config = MRI2PETConfig()
mri_list = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))
random.shuffle(mri_list)

for mri_fpath in tqdm(mri_list):
    fpath = mri_fpath.replace('.nii.gz', '.npy')
    if os.path.exists(fpath):
        continue

    img = ants.image_read(mri_fpath)
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())
    np.save(fpath, data)