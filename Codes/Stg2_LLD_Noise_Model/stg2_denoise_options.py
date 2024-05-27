

import argparse
import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="DResTCN_Gray")

parser.add_argument("--trainset_path", type=str, default='../dataset/denoising/sid_raw/Train_Paired_RGBG_p512_N4480_FPN', help="path to train set")
parser.add_argument("--train_dir", type=str, default='../dataset/denoising/', help="path to train set")
parser.add_argument("--eval_dir", type=str, default='../dataset/SID/Sony', help="path to train set")
# parser.add_argument("--trainset_path", type=str, default="/share/data/cy/trainset_p50_s10_rgb.h5", help="path to train set")
# parser.add_argument("--trainset_path", type=str, default="/share/data/cy/trainset_p120_s200_gray.h5", help="path to train set")
# local
# parser.add_argument("--trainset_path", type=str, default="./h5_files/trainset_gray.h5", help="path to train set")







# Validation Set
# server
# parser.add_argument("--valset_path", type=str, default="/home/xhwu/lowlevel/DilatedTCN/data_val/", help="path to val set")
#
parser.add_argument("--patch_size", type=int, default=512, help="the patch size of input")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--load_thread", type=int, default=0, help="thread for data loader")
# net

# save
parser.add_argument("--last_ckpt",type=str,default="/dn_raw_DnCNN_syn_e47.pth",help="the ckpt of last net")
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
parser.add_argument("--save_prefix", type=str, default="dn_raw_DnCNN_syn_e",help="prefix added to all ckpt to be saved")
parser.add_argument("--log_dir", type=str, default='./logs_s1', help='path of log files')
parser.add_argument("--save_every", type=int, default=100, help="Number of training steps too log psnr and perform")
parser.add_argument("--save_every_epochs", type=int, default=1, help="Number of training epchs to save state")

parser.add_argument("--learning_rate_dtcn", type=float, default=1e-4, help="the initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.5, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs the model needs to run")
parser.add_argument("--steps", type=str, default="100,180", help="schedule steps,use comma(,) between numbers")

parser.add_argument("--save_path", type=str, default='./denoise_last_ckpt/',help="prefix added to all ckpt to be saved")

opt = parser.parse_args()

# Normalize noise between [0, 1]
steps = opt.steps
steps = steps.split(',')
opt.steps = [int(eval(step)) for step in steps]


