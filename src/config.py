class MRI2PETConfig(object):
    def __init__(
            self,            
            batch_size=48,
            epoch=5000,
            pretrain_epoch=500,
            patience=5,
            lr=1e-4,
            mri_image_dim=168,
            pet_image_dim=128,
            n_mri_channels=128,
            n_pet_channels=60,
            embed_dim = 128,
            beta_start = 0.0015,
            beta_end = 0.02,
            num_timesteps = 1000,
            z_dim = 128,
            lambda_gp = 10,
            generator_interval = 2,
            mri_pretrain_dir = "/data/CARD_AA/data/ADNI/MRI_Pretrain/",
            mri_style_dir = "/data/CARD_AA/data/ADNI/MRI_StyleTransfer/",
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
        self.mri_pretrain_dir = mri_pretrain_dir
        self.mri_style_dir = mri_style_dir

# PET Heights: Counter({128: 2977})
# PET Channels: Counter({60: 2977})
# MRI Heights: Counter({176: 3355, 160: 2121, 166: 1837, 196: 1720, 170: 1580, 208: 1394, 384: 997, 180: 778, 352: 616, 172: 537, 173: 535, 171: 513, 174: 484, 175: 473, 256: 453, 168: 431, 169: 427, 177: 423, 167: 373, 179: 352, 211: 334, 178: 326, 181: 309, 165: 284, 164: 247, 182: 188, 163: 184, 184: 154, 183: 147, 162: 126, 161: 112, 187: 93, 185: 86, 186: 78, 188: 76, 340: 70, 189: 70, 191: 49, 376: 46, 190: 46, 159: 37, 192: 34, 1024: 33, 158: 29, 496: 28, 194: 25, 193: 24, 512: 23, 200: 21, 272: 21, 195: 21, 304: 19, 422: 15, 157: 15, 199: 14, 198: 14, 201: 13, 124: 12, 230: 12, 204: 9, 202: 9, 156: 9, 205: 9, 155: 9, 203: 8, 154: 7, 197: 7, 207: 5, 320: 5, 206: 5, 288: 4, 350: 4, 576: 3, 144: 2, 140: 2, 382: 2, 152: 2, 150: 1, 146: 1, 128: 1, 145: 1, 149: 1, 142: 1, 368: 1, 386: 1, 328: 1, 292: 1, 432: 1, 494: 1, 232: 1, 248: 1, 212: 1, 308: 1, 276: 1, 236: 1, 640: 1, 348: 1, 151: 1})
# MRI Channels: Counter({256: 9957, 192: 1916, 512: 1841, 170: 392, 200: 281, 201: 261, 199: 255, 197: 253, 198: 250, 203: 249, 196: 247, 204: 242, 205: 235, 202: 234, 195: 228, 206: 221, 191: 217, 193: 216, 194: 215, 189: 212, 209: 206, 207: 199, 208: 194, 190: 192, 185: 182, 188: 177, 210: 175, 211: 172, 184: 164, 186: 158, 213: 157, 187: 157, 212: 156, 214: 154, 215: 141, 217: 126, 182: 124, 216: 122, 218: 120, 183: 111, 219: 101, 181: 88, 221: 85, 179: 85, 220: 83, 180: 76, 223: 68, 226: 68, 222: 63, 224: 63, 225: 59, 178: 59, 227: 54, 228: 47, 229: 38, 177: 37, 175: 36, 231: 35, 233: 34, 230: 32, 576: 30, 232: 30, 176: 29, 239: 27, 384: 26, 174: 25, 234: 24, 251: 23, 240: 22, 244: 20, 235: 20, 238: 20, 237: 20, 243: 18, 252: 17, 248: 17, 249: 16, 236: 16, 241: 16, 246: 15, 173: 15, 247: 14, 245: 14, 250: 12, 254: 12, 169: 12, 253: 11, 1024: 10, 255: 10, 172: 10, 242: 10, 168: 8, 171: 8, 272: 5, 352: 4, 156: 4, 288: 3, 160: 3, 165: 3, 332: 3, 360: 3, 164: 3, 320: 3, 167: 3, 144: 2, 54: 2, 260: 2, 152: 2, 161: 2, 150: 1, 416: 1, 312: 1, 328: 1, 44: 1, 340: 1, 266: 1, 350: 1, 151: 1, 166: 1, 155: 1, 163: 1})
# MRI Paths: 22956
# PET-MRI Pairs: 2772
