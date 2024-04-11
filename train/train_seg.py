# -*- coding : UTF-8 -*-
# @file   : train_seg.py
# @Time   : 2024-03-28 17:01
# @Author : wmz
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import sys
from models.Unet3D import UNet_3D as UNet
from utils.filters import Gaussian
from utils.metric import Dice
from utils.utils import *
from dataset.CommonDataSet import CommonDataset
import time
import os


def set_logger(MODEL_PATH: str):
    '''设置日志'''
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = "{}/log.txt".format(MODEL_PATH)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_seg(args):
    ckpt = args.ckpt
    device = args.device if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    batch_size = args.batch_size
    number_class = args.num_class
    pretrained = args.pretrained
    base_lr = args.lr
    root_dir = args.root
    image_suffix = args.img_suffix
    save_name = str(batch_size)
    save_path = args.log_path
    model_path = args.model
    best_score = 0
    
    os.makedirs(f"{model_path}", exist_ok=True)
    logger = set_logger(model_path)
    writer = SummaryWriter(save_path + save_name)
    train_file_list = get_files(os.path.join(root_dir, 'train'), image_suffix)
    train_set = CommonDataset(root_dir, train_file_list, 'train')
    test_file_list = get_files(os.path.join(root_dir, 'test'), image_suffix)
    test_set = CommonDataset(root_dir, test_file_list, 'test')
    val_file_list = get_files(os.path.join(root_dir, 'val'), image_suffix)
    val_set = CommonDataset(root_dir, val_file_list, 'val')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    model = UNet(2, number_class).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = MultiStepLR(optimizer, milestones=[epochs // 3, epochs // 3 * 2], gamma=0.1)
    gaussian = Gaussian(3, None, 5, norm=True).to(device)
    start_epoch = 0
    if pretrained:
        model.load_state_dict(torch.load(pretrained)['model_state_dict'])

    if ckpt:
        checkpoint = torch.load(ckpt)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['score']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Start training")
    for ep in tqdm(range(start_epoch, epochs)):
        tqdm.write(f"----------{ep}----------")
        model.train()
        train_loss = 0.
        train_dc = 0.
        training_all_loss = 0.0
        acc = 0
        num_batches = 0
        begin = time.time()
        for _, (ID, img_path) in enumerate(train_loader):
            patches = train_set.generate_train_patch(img_path[0])
            patch_loader = DataLoader(patches, 1)
            for i, (image, mask, landmark) in enumerate(patch_loader):
                num_batches += 1
                optimizer.zero_grad()
                image = image.to(device)
                mask = mask.to(device)
                heatmap = gaussian(landmark.to(device))
                inputs = torch.cat([image, heatmap], dim=1)
                output = model(inputs)
                output = torch.sigmoid(output)
                loss = loss_func(output, mask.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                output = output.squeeze().cpu().detach().numpy()
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                mask = mask.squeeze().cpu().numpy()
                dc = Dice(output, mask)
                train_dc += dc
                print(
                    f"Ep:{ep + 1}\tID:{ID[0]}\tLoss:{loss.item():.6f}\tDice:{dc * 100:.2f}%", end='\r')
        end = time.time()
        train_loss /= num_batches
        train_dc /= num_batches
        logger.info(
                    f"Epoch:{ep + 1}/{epochs}\ttrain_loss:{train_loss:.6f}\ttrain_dice:{train_dc * 100:.2f}%\tTime:{end - begin:.3f}s"
                )
        # save checkpoint every epoch
        checkpoint = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": best_score,
            "loss": train_loss,
        }
        torch.save(checkpoint, f"{model_path}/checkpoint.pth")

        if train_dc > best_score:
            best_score = train_dc
            torch.save(checkpoint, f"{model_path}/best_checkpoint.pth")


if __name__ == "__main__":
    setup_seed(33)
    parser = argparse.ArgumentParser(description='Segmentation training')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--root", type=str, default='E:/Data/VerSe')
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--img_suffix", type=str, default="_ct.nii.gz")
    parser.add_argument('-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-epochs', default=10, type=int, help='training epochs')
    parser.add_argument('-eval_epoch', default=1, type=int, help='evaluation epoch')
    parser.add_argument('-log_path', default="logs", type=str, help='the path of the log')
    parser.add_argument('-read_params', default=False, type=bool, help='if read pretrained params')
    parser.add_argument('-params_path', default="", type=str, help='the path of the pretrained model')
    parser.add_argument('-basepath', default="", type=str, help='base dataset path')
    parser.add_argument('-augmentation', default=False, type=bool, help='if augmentation')
    parser.add_argument('-num_class', default=1, type=int, help='the number of class')
    parser.add_argument("--model", type=str, default='unet_mse')
    parser.add_argument('-model_name', default="", type=str, help='Unet3D')

    args = parser.parse_args()
    train_seg(args)


