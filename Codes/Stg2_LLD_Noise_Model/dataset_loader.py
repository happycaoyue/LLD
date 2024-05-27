import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_mat_file, is_png_file, load_raw_mat, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
# from models.modules.LLD_DN_arch import LLD_DN_arch
import torch.nn as nn
import natsort
import rawpy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class SID_Dataset_Denoise_raw(Dataset):
    def __init__(self, dataset_dir, patchsize=None, train_flag = True):
        super(SID_Dataset_Denoise_raw, self).__init__()
        #gt_dir = 'groundtruth'
        #clean_files = sorted(os.listdir(os.path.join(dataset_dir, gt_dir)))
        #self.clean_filenames = [os.path.join(dataset_dir, gt_dir, x) for x in clean_files if is_mat_file(x)]
        self.patchsize = patchsize

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(dataset_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataset_dir, input_dir)))

        import natsort
        clean_files = natsort.natsorted(os.listdir(os.path.join(dataset_dir, gt_dir)))
        noisy_files = natsort.natsorted(os.listdir(os.path.join(dataset_dir, input_dir)))

        self.clean_filenames = [os.path.join(dataset_dir, gt_dir, x) for x in clean_files if is_mat_file(x)]
        self.noisy_filenames = [os.path.join(dataset_dir, input_dir, x) for x in noisy_files if is_mat_file(x)]

        self.tar_size = len(self.clean_filenames)


    def random_sample_iso(self):
        index = random.randint(0, self.isos_length)
        return self.isos[index]

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        # print(self.tar_size)
        mat_clean = load_raw_mat(self.clean_filenames[tar_index])

        mat_noisy = load_raw_mat(self.noisy_filenames[tar_index])

        # (4, 512, 512)
        iso = np.float32(mat_clean['ISO'])[0][0]
        ratio = np.float32(mat_clean['ratio'])[0][0]
        isos = [50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200,
                4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600]

        clean_np = np.float32(np.array(mat_clean['Iclean_crop']))
        noisy_np = np.float32(np.array(mat_noisy['Inoisy_crop']))
        clean = clean_np.astype(np.float32)
        clean = np.clip(clean * ratio, 0., 1.)
        short_noisy_channels = np.clip(noisy_np * ratio, 0., 1.)

        clean = clean
        noisy = short_noisy_channels
        noisy = np.clip(noisy, 0., 1.)
        noisy = torch.FloatTensor(noisy.copy())
        clean = torch.FloatTensor(clean.copy())

        iso_torch = torch.FloatTensor(np.expand_dims(iso, axis=0))
        ratio_torch = torch.FloatTensor(np.expand_dims(ratio, axis=0))

        return {'clean':clean, 'noisy':noisy, 'ratio':ratio_torch, 'ISO':iso_torch}

