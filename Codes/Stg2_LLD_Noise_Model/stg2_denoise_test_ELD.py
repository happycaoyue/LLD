#
import os
import argparse
import random
import datetime
from pathlib import Path
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
import dataset.sid_dataset as datasets
import dataset.lmdb_dataset as lmdb_dataset
import dataset
from dataset.sid_dataset import worker_init_fn
from net.UNetSeeInDark import Net
import util.util as util
from data_process import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def findLastCheckpoint(save_dir, save_pre):
    file_list = glob.glob(os.path.join(save_dir, save_pre + '*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*" + save_pre +"(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i + 1, ...] = self.correct(predict[i:i + 1, ...], source[i:i + 1, ...])
            else:
                for i in range(predict.shape[0]):
                    output[i:i + 1, ...] = self.correct(predict[i:i + 1, ...], source)
        else:
            output = self.correct(predict, source)
        return output
    # predict = [(predict * source) / (predict * predict)] * predict
    def correct(self, predict, source):
        N, C, H, W = predict.shape
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)
        output = num / den * predict
        return output
def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.color_matrix[:3, :3].astype(np.float32)
    if ccm[0,0] == 0:
        ccm = np.eye(3, dtype=np.float32)
    return wb, ccm
def valid(args):
    # torch.set_num_threads(4)
    # new or continue
    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre = args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'

    if args.resume == "continue":
        # load  validation set
        databasedir = '../dataset/ELD_new'
        cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
        suffixes = ['.CR2', '.CR2', '.CR2', '.nef', '.ARW']
        # scene-1 ~ scene-10
        include = 4
        scenes = list(range(1, 10 + 1))
        cameras = cameras[include:include + 1]
        suffixes = suffixes[include:include + 1]


        # net architecture
        dn_net = Net()
        alpha = IlluminanceCorrect()
        # Move to GPU
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dn_model = nn.DataParallel(dn_net).cuda()
        dn_model = nn.DataParallel(dn_net).cuda()
        alpha = nn.DataParallel(alpha).cuda()

    if args.resume == "continue":
        tmp_ckpt=torch.load(args.last_ckpt)

        start_epoch = initial_epoch + 1

        # Initialize dn_model
        pretrained_dict = tmp_ckpt['state_dict']
        model_dict = dn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dn_model.load_state_dict(model_dict)

    if args.resume == "continue":
        print("---------------------start valid experiment-------------------------------")
        dn_model.eval()
        s = sio.loadmat('test_epoch_psnr_dncnn.mat')
        psnr_data = s["tep"]
        psnr_all = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        ssim_all = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        img_ids_set = [[4, 9, 14], [5, 10, 15]]
        # res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=False, crop=False)
        jj = 0
        for i, img_ids in enumerate(img_ids_set):
            eval_datasets = [datasets.ELDEvalDataset(databasedir, camera_suffix, scenes=scenes, img_ids=img_ids) for
                             camera_suffix in zip(cameras, suffixes)]

            eval_dataloaders = [torch.utils.data.DataLoader(
                eval_dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True) for eval_dataset in eval_datasets]

            for camera, val_loader in zip(cameras, eval_dataloaders):
                psnr_val = 0
                ssim_val = 0
                psnr_val_alpha = 0
                ssim_val_alpha = 0
                count = 0.0
                for i, data in enumerate(val_loader):
                    count = count + 1.0
                    clean = data['target'].cuda()
                    noisy = data['input'].cuda()
                    with torch.no_grad():
                        out = dn_model(noisy)
                        output_alpha = alpha(out, clean)
                    imgs_dn = torch.clamp(output_alpha, 0, 1)
                    psnr = batch_psnr(out.clamp(0., 1.), clean.clamp(0., 1.), 1.)
                    psnr_val += psnr
                    psnr = batch_psnr(output_alpha.clamp(0., 1.), clean.clamp(0., 1.), 1.)
                    psnr_val_alpha += psnr

                    out_numpy = out[0].cpu().clamp(0., 1.).float().numpy()
                    out_numpy = (np.transpose(out_numpy, (1, 2, 0))) * 255.0

                    output_alpha_numpy = output_alpha[0].cpu().clamp(0., 1.).float().numpy()
                    output_alpha_numpy = (np.transpose(output_alpha_numpy, (1, 2, 0))) * 255.0

                    clean_numpy = clean[0].cpu().float().numpy()
                    clean_numpy = (np.transpose(clean_numpy, (1, 2, 0))) * 255.0

                    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
                    ssim = structural_similarity(out_numpy, clean_numpy, data_range=255, multichannel=True)
                    ssim_val += ssim
                    ssim = structural_similarity(output_alpha_numpy, clean_numpy, data_range=255, multichannel=True)
                    ssim_val_alpha += ssim
                    SonyCCM = np.array([[1.9712269, -0.6789218, -0.29230508],
                                        [-0.29104823, 1.748401, -0.45735288],
                                        [0.02051281, -0.5380369, 1.5175241]])

                    # data['wb'] = self.infos[scene_id][hr_id]['wb']
                    # data['ccm'] = self.infos[scene_id][hr_id]['ccm']
                    target_path = data['rawpath'][0]
                    input_path = data['input_path'][0]
                    # print(target_path)
                    name = input_path.split('/')[-2] + "_" + input_path.split('/')[-1][:-4] + "_OurNM_"
                    raw = rawpy.imread(target_path)
                    wb, ccm = read_wb_ccm(raw)
                    output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=SonyCCM)
                    # print(output.shape)
                    save_path = "./OurNM_image/"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    psnr_str = "%.2f" % psnr
                    ssim_str = "%.4f" % ssim
                    psnr_str = psnr_str.replace('.', '_')
                    ssim_str = ssim_str.replace('.', '_')
                    # print(name)
                    # print(psnr_str)
                    # print(ssim_str)
                    filename = name + psnr_str + "_" + ssim_str + ".png"
                    print(filename)
                    denoisedfile = os.path.join(save_path, filename)

                    cv2.imwrite(denoisedfile, output[:, :, ::-1])
                psnr_all[jj] = psnr_val / count
                psnr_all[jj+2] = psnr_val_alpha / count
                ssim_all[jj] = ssim_val / count
                ssim_all[jj+2] = ssim_val_alpha / count
                jj = jj + 1

        psnr_data = np.column_stack((psnr_data,
                                     [start_epoch-1, psnr_all[0], ssim_all[0], psnr_all[1], ssim_all[1],
                                      psnr_all[2], ssim_all[2], psnr_all[3], ssim_all[3],
                                      psnr_all[4], ssim_all[4], psnr_all[5], ssim_all[5]]))
        s["tep"] = psnr_data
        sio.savemat('test_epoch_psnr_dncnn.mat', s)
