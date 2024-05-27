#
import os
import random
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
from dataset_loader import SID_Dataset_Denoise_raw
from stg2_denoise_options import opt
from net.UNetSeeInDark import Net
random.seed()
import stg2_denoise_test_SID
import stg2_denoise_test_ELD
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def main(args):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('test_epoch_psnr_dncnn.mat'):
        s = {}
        s["tep"] = np.zeros((13, 1))
        sio.savemat('test_epoch_psnr_dncnn.mat', s)
    # new or continue
    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre = args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'

    # net architecture
    dn_net = Net()

    # loss function
    criterion = nn.L1Loss().cuda()
    # Move to GPU
    # dilated_tcn_model = nn.DataParallel(dilated_tcn_net).cuda()
    if torch.cuda.device_count() > 1:
        dn_model = nn.DataParallel(dn_net).cuda()
    else:
        dn_model = nn.DataParallel(dn_net).cuda()
    # Optimizer
    training_params = None
    optimizer_dn = None

    # load old params, optimizer, state
    if args.resume == "continue":
        tmp_ckpt = torch.load(args.last_ckpt)

        start_epoch = initial_epoch + 1

        # Initialize dn_model
        pretrained_dict = tmp_ckpt['state_dict']
        model_dict=dn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dn_model.load_state_dict(model_dict)
        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dtcn)
        optimizer_dn.load_state_dict(tmp_ckpt['optimizer_state'])

    elif args.resume == "new":
        training_params = {}
        training_params['step'] = 1
        start_epoch = 1

        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dtcn)
    if args.resume=="continue":
        # test SID
        stg2_denoise_test_SID.valid(args)
        # test ELD
        stg2_denoise_test_ELD.valid(args)

    # set training set DataLoader
    train_dataset = SID_Dataset_Denoise_raw(args.trainset_path, patchsize=args.patch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.load_thread, pin_memory=True, drop_last=False)

    # training
    for epoch in range(start_epoch, args.epoch+1):
        dn_model.train()
        # train
        i = 0
        total_step = len(train_loader)
        lr_s = 1e-4
        if epoch >= 100:
            lr_s = 5e-5
        if epoch >= 180:
            lr_s = 1e-5
        if epoch == start_epoch or epoch == 100 or epoch == 180:
            # for optimizer in optimizer_dn:
            for group in optimizer_dn.param_groups:
                group['lr'] = lr_s
        for i, data in enumerate(train_loader, 0):
            img_gt = data['clean'].cuda()
            img_noise = data['noisy'].cuda()
            optimizer_dn.zero_grad()

            batch, C, H, W = img_noise.size()
            if batch == args.batch_size:
                img_output = dn_model(img_noise)
                loss = criterion(img_gt, img_output)
                loss.backward()
                optimizer_dn.step()
                i = i + 1
                print("Epoch:[{}/{}] Batch: [{}/{}] loss = {:.4f}".format(epoch, args.epoch, i, total_step, loss.item()))

        if epoch % args.save_every_epochs == 0:
            # save model and checkpoint
            save_dict = {'state_dict': dn_model.state_dict(),
                        'optimizer_state': optimizer_dn.state_dict()}
            torch.save(save_dict, os.path.join(args.save_path + args.save_prefix + '{}.pth'.format(epoch)))
            del save_dict
            # test SID
            stg2_denoise_test_SID.valid(args)
            # test ELD
            stg2_denoise_test_ELD.valid(args)

if __name__ == "__main__":

    main(opt)

    exit(0)



