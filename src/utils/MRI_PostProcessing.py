import ants
import pickle
import numpy as np
from tqdm import tqdm
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()
mri_list = pickle.load(open('./src/data/mriDataset.pkl', 'rb'))

for mri_fpath in tqdm(mri_list):
    img = ants.image_read(fpath)
    data = img.numpy()
    data = (data - data.min()) / (data.max() - data.min())
    fpath = mri_fpath.replace('.nii.gz', '.npy')
    np.save(fpath, data)