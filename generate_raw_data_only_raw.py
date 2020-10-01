"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.cycleisp import Rgb2Raw
from dataloaders.data_rgb import get_rgb_data
from utils.noise_sampling import random_noise_levels_dnd, random_noise_levels_sidd, add_noise
import utils
# import lycon
import cv2
import glob
import math
import gc

parser = argparse.ArgumentParser(description='RGB2RAW Network: From clean RGB images, generate {RAW_clean, RAW_noisy} pairs')
parser.add_argument('--input_dir', default=None,
    type=str, help='Directory of clean RGB images')
parser.add_argument('--list_from_file', default=None,
    type=str, help='image list of clean RGB images')
parser.add_argument('--result_dir', default='./results/synthesized_data/raw/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/isp/rgb2raw.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save synthesized images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(os.path.join(args.result_dir, 'pkl'))
utils.mkdir(os.path.join(args.result_dir,'png'))

# test_dataset = get_rgb_data(args.input_dir)
# test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

if (args.input_dir) and (args.list_from_file):
    exit('[ERROR] Only either --input_dir or --list_from_file can be activate')

img_list = []
if args.input_dir:
    img_list = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')))
elif args.list_from_file:
    with open(args.list_from_file, 'r') as f:
        for i in f.readlines():
            img_list.append(i.rstrip())
print(f'There are {len(img_list)} images to be processed.')


model_rgb2raw = Rgb2Raw()

utils.load_checkpoint(model_rgb2raw,args.weights)
print("===>Testing using weights: ", args.weights)

model_rgb2raw.cuda()

model_rgb2raw=nn.DataParallel(model_rgb2raw)

model_rgb2raw.eval()


crop_size=512
min_crop_size = 32

with torch.no_grad():
    for path in tqdm(img_list):
        filename = os.path.basename(path)
        # img = lycon.load(path)
        img = cv2.imread(path)[:,:,::-1]
        img = img.astype(np.float32)
        img = img/255.
        img_h, img_w, _ = img.shape
        print(f'{filename} image input shape={img.shape}')
        tile_output = torch.zeros((4, img_h // 2, img_w // 2))

        shift_X = shift_Y = crop_size

        for x in range(0, img_w, shift_X):
            for y in range(0, img_h, shift_Y):
                X_upper = min(x + shift_X, img_w)
                Y_upper = min(y + shift_Y, img_h)
                X_lower = max(0, X_upper-shift_X)
                Y_lower = max(0, Y_upper-shift_Y)

                input_img = np.zeros((crop_size, crop_size,3))
                size_Y = Y_upper - Y_lower
                size_X = X_upper - X_lower

                input_img[:size_Y,:size_X,:] = img[Y_lower:Y_upper, X_lower:X_upper, :]
                rgb_gt = torch.from_numpy(input_img).float()
                rgb_gt = rgb_gt.permute(2,0,1)
                rgb_gt = rgb_gt.unsqueeze(0).to('cuda')

                ## Convert clean rgb image to clean raw image
                raw_gt = model_rgb2raw(rgb_gt)       ## raw_gt is in RGGB format
                raw_gt = torch.clamp(raw_gt,0,1)


                raw_gt = raw_gt.squeeze(0).cpu().detach()
                tile_output[:, Y_lower//2:Y_upper//2, X_lower//2:X_upper//2] = raw_gt[:,:size_Y // 2,:size_X//2]

        #### Unpadding and saving
        print(f'output shape={tile_output.shape}')
        clean_packed = tile_output[:,:,:]   ## RGGB channels  (4 x H/2 x W/2)
        clean_unpacked = utils.unpack_raw(clean_packed.unsqueeze(0))                                  ## Rearrange RGGB channels into Bayer pattern
        clean_unpacked = clean_unpacked.squeeze().cpu().detach().numpy()

        try:
            print(os.path.join(args.result_dir, 'png', filename[:-4]+'.png'))
            # lycon.save(os.path.join(args.result_dir, 'png', filename[:-4]+'.png'),(clean_unpacked*255).astype(np.uint8))
            cv2.imwrite(os.path.join(args.result_dir, 'png', filename[:-4]+'.png'),(clean_unpacked*255).astype(np.uint8))
            # cv2.imwrite(args.result_dir+'png/clean/'+filename[:-4]+'.png',(clean_unpacked*255).astype(np.uint8))
        except cv2.error as e:
            print(filename)
            print(clean_packed)
            #import pdb;pdb.set_trace()
        dict_ = {}
        dict_['raw'] = clean_packed.cpu().detach().numpy()       ## (4 x H/2 x W/2)
        utils.save_dict(dict_, os.path.join(args.result_dir, 'pkl', filename[:-4]+'.pkl'))
        # gc.collect()

