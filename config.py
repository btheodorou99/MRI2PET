class MRI2PETConfig(object):
    def __init__(
            self,            
            batch_size=64,
            epoch=500,
            pretrain_epoch=500,
            patience=5,
            lr=1e-4,
            mri_image_dim=256,
            pet_image_dim=256,
            n_mri_channels=10,
            n_pet_channels=50,
            embed_dim = 128,
            beta_start = 0.0015,
            beta_end = 0.02,
            num_timesteps = 1000,
            z_dim = 128,
            lambda_gp = 10,
            generator_interval = 2,
            image_dir = '/srv/local/data/MIMIC-CXR/images',
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.pretrain_epoch = pretrain_epoch
        self.patience = patience
        self.lr = lr
        self.mri_image_dim = mri_image_dim
        self.pet_image_dim = pet_image_dim
        self.n_mri_channels = n_mri_channels
        self.n_pet_channels = n_pet_channels
        self.embed_dim = embed_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.generator_interval = generator_interval
        self.image_dir = image_dir