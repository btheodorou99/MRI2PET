import ants
import pickle
import numpy as np
from tqdm import tqdm
import nibabel as nib
from ..config import MRI2PETConfig

config = MRI2PETConfig()
mri_list = pickle.load(open('./src/data/original_mriDataset.pkl', 'rb'))

def convert_mri_path(fpath):
    fpath = fpath.replace('CARDPB', 'CARD_AA')
    start, end = fpath.split('MRI')
    fname = end.split('/')[-1]
    return start + 'MRI/' + fname

def process_mri(fpath):
    img = ants.image_read(fpath)
    img = ants.from_numpy(np.flip(img.numpy(), axis=2))
    img = ants.resample_image(img, (config.mri_image_dim, config.mri_image_dim, config.n_mri_channels), use_voxels=True, interp_type=3)
    return img

for mri_fpath in tqdm(mri_list):
    img = process_mri(mri_fpath)
    fpath = convert_mri_path(mri_fpath)
    img = ants.utils.convert_nibabel.to_nibabel(img)
    nib.save(img, fpath)