import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, config, is_mri=True):
        super(ImageEncoder, self).__init__()
        self.depth = config.n_mri_channels if is_mri else config.n_pet_channels
        self.image_dim = config.mri_image_dim if is_mri else config.pet_image_dim

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flat_dim = 32 * (self.depth // 8) * (self.image_dim // 8) * (self.image_dim // 8)
        self.fc1 = nn.Linear(self.flat_dim, config.embed_dim)
        self.fc2 = nn.Linear(config.embed_dim, config.embed_dim)
        
    def forward(self, x):        
        # Ready for 3D Convolution Input
        x = x.unsqueeze(1)

        # Convolution + ReLU + MaxPooling
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)

        # Flattening the output
        x = x.view(-1, self.flat_dim)

        # Passing through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ImageClassifier(nn.Module):
    def __init__(self, config, has_mri=False, has_pet=False):
        super(ImageClassifier, self).__init__()
        assert has_pet or has_mri, "At least one of MRI or PET must be present"
        self.has_mri = has_mri
        self.has_pet = has_pet
        if has_mri:
            self.mri_encoder = ImageEncoder(config, is_mri=True)
        if has_pet:
            self.pet_encoder = ImageEncoder(config, is_mri=False)

        self.fc1 = nn.Linear(2*config.embed_dim if has_pet and has_mri else config.embed_dim, config.embed_dim)
        self.fc2 = nn.Linear(config.embed_dim, config.downstream_dim)
        
    def forward(self, mri, pet):
        if self.has_mri:
            mri = self.mri_encoder(mri)
        if self.has_pet:
            pet = self.pet_encoder(pet)
        x = torch.cat([mri, pet], dim=1) if self.has_pet and self.has_mri else mri if self.has_mri else pet
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImageRegressor(nn.Module):
    def __init__(self, config, has_mri=True, has_pet=True):
        super().__init__()
        assert has_pet or has_mri, "At least one of MRI or PET must be present"
        self.has_mri = has_mri
        self.has_pet = has_pet
        if has_mri:
            self.mri_encoder = ImageEncoder(config, is_mri=True)
        if has_pet:
            self.pet_encoder = ImageEncoder(config, is_mri=False)

        self.regressor = nn.Sequential(
            nn.Linear(2*config.embed_dim if has_pet and has_mri else config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.embed_dim, 1)  # Single output for MMSE score
        )
    
    def forward(self, mri, pet):
        if self.has_mri:
            mri = self.mri_encoder(mri)
        if self.has_pet:
            pet = self.pet_encoder(pet)
        x = torch.cat([mri, pet], dim=1) if self.has_pet and self.has_mri else mri if self.has_mri else pet
        return self.regressor(x).squeeze()