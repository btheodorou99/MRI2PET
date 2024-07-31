import os
import ants
from tqdm import tqdm
from collections import Counter

input_dir = "/data/CARD_AA/data/ADNI/PET_Nifti/"
output_dir = "/data/theodoroubp/MRI2PET/results/data_exploration/"
os.makedirs(output_dir, exist_ok=True)

depths = []
heights = []
widths = []

for f in tqdm(os.listdir(input_dir)):
    if not f.endswith(".nii") or os.path.exists(os.path.join(output_dir, f.replace(".nii", ".png"))):
        continue

    # print(f)
    img = ants.image_read(os.path.join(input_dir, f))
    # print(img.shape)
    if len(img.shape) == 4:
        if f == '037_S_4015--2013-04-19--I368085.nii':
            print(img[:, :, :, 0].max())
            img = ants.from_numpy(img[:, :, :, 1])
        else:
            img = ants.from_numpy(img[:, :, :, 0])

    img = ants.from_numpy(img.numpy().transpose(2, 0, 1))
    depths.append(img.shape[0])
    heights.append(img.shape[1])
    widths.append(img.shape[2])
    ants.plot(
        img,
        nslices=9,
        title=str(img.shape[0]),
        filename=f"{output_dir}{f.replace('.nii', '.png')}",
    )

print(Counter(depths))
print(Counter(heights))
print(Counter(widths))
