class MRI2PETConfig(object):
    def __init__(
            self,            
            batch_size=50,
            epoch=2500,
            pretrain_epoch=500,
            lr=1e-4,
            mri_image_dim=144,
            pet_image_dim=128,
            n_mri_channels=64,
            n_pet_channels=32,
            embed_dim = 256,
            beta_start = 0.0015,
            beta_end = 0.02,
            num_timesteps = 1000,
            laplace_lambda = 0.25,
            downstream_dim = 1,
            downstream_batch_size = 128,
            downstream_epoch = 100,
            downstream_patience = 5,
            downstream_lr = 5e-4,
            z_dim = 256,
            lambda_gp = 10,
            generator_interval = 2,
            n_bootstrap = 100,
            mri_pretrain_dir = "/data/CARD_AA/data/ADNI/MRI_Pretrain/",
            mri_style_dir = "/data/CARD_AA/data/ADNI/MRI_StyleTransfer/",
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.pretrain_epoch = pretrain_epoch
        self.lr = lr
        self.mri_image_dim = mri_image_dim
        self.pet_image_dim = pet_image_dim
        self.n_mri_channels = n_mri_channels
        self.n_pet_channels = n_pet_channels
        self.embed_dim = embed_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.laplace_lambda = laplace_lambda
        self.downstream_dim = downstream_dim
        self.downstream_batch_size = downstream_batch_size
        self.downstream_epoch = downstream_epoch
        self.downstream_patience = downstream_patience
        self.downstream_lr = downstream_lr
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.generator_interval = generator_interval
        self.n_bootstrap = n_bootstrap
        self.mri_pretrain_dir = mri_pretrain_dir
        self.mri_style_dir = mri_style_dir

# MRI Paths: 22956
# PET-MRI Pairs: 2767
