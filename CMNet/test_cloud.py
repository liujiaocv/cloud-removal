"""

"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from CMNet import CMNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Cloud removal using CMNet')

parser.add_argument('--input_dir', default='/home/liujiao/code/cloud/T-Cloud/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results_tcloud/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/liujiao/code/cloud/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = CMNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

rgb_dir_test = os.path.join(args.input_dir,  'input')
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

utils.mkdir(args.result_dir)

with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_    = data_test[0].cuda()
            filenames = data_test[1]

            restored = model_restoration(input_)
            restored = torch.clamp(restored[0],0,1)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(args.result_dir, filenames[batch]+'.png')), restored_img)






