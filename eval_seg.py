# -*- coding : UTF-8 -*-
# @file   : eval_seg.py
# @Time   : 2024-03-28 17:51
# @Author : wmz
import SimpleITK as sitk
import sys
import torch
from torch.utils.data import DataLoader
from models.Unet3D import UNet_3D as UNet
from utils.filters import Gaussian
from utils.metric import Dice
from utils.utils import *
from dataset.CommonDataSet import CommonDataset
import time
import os
import pandas as pd
import numpy as np


if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    root_dir = "E:/Dataset/VerSe19"
    image_suffix = "_ct.nii.gz"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    train_file_list = get_files(os.path.join(root_dir, 'train'), image_suffix)
    train_set = CommonDataset(root_dir, train_file_list, 'val')
    test_file_list = get_files(os.path.join(root_dir, 'test'), image_suffix)
    test_set = CommonDataset(root_dir, test_file_list, 'val')
    val_file_list = get_files(os.path.join(root_dir, 'validation'), image_suffix)
    val_set = CommonDataset(root_dir, val_file_list, 'val')
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    model = UNet(2, 1).to(device)
    ckpt = torch.load("unet_mse/best_checkpoint.pth", map_location=device)
    print(ckpt['score'])
    model.load_state_dict(ckpt['model_state_dict'])
    gaussian = Gaussian(3, None, 5, norm=True).to(device)
    id_list = []
    dc_list = []
    dataset_list = []
    category_list = []
    time_list = []

    for _, (ID, img_path, classes) in enumerate(train_loader):
        patches = train_set.generate_inf_patch(img_path[0])
        patch_loader = DataLoader(patches, batch_size)
        classes = torch.cat(classes)
        classes = classes.tolist()
        for i, (new_img, new_mask, landmark) in enumerate(patch_loader):
            new_img = new_img.to(device)
            heatmap = gaussian(landmark.to(device))
            inputs = torch.cat([new_img, heatmap], dim=1)
            begin = time.time()
            with torch.no_grad():
                pred = model(inputs)
            new_mask = new_mask.squeeze().numpy()
            pred = pred.cpu().squeeze().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            end = time.time()
            dc = Dice(pred, new_mask)
            id_list.append(ID[0])
            dc_list.append(dc)
            dataset_list.append('train')
            category_list.append(classes[i])
            time_list.append(end - begin)
            print(f"{ID[0]}\t{classes[i]}\t{dc * 100:.2f}%")

    for _, (ID, img_path, classes) in enumerate(val_loader):
        patches = val_set.generate_inf_patch(img_path[0])
        patch_loader = DataLoader(patches, batch_size)
        classes = torch.cat(classes)
        classes = classes.tolist()
        for i, (new_img, new_mask, landmark) in enumerate(patch_loader):
            new_img = new_img.to(device)
            heatmap = gaussian(landmark.to(device))
            inputs = torch.cat([new_img, heatmap], dim=1)
            begin = time.time()
            with torch.no_grad():
                pred = model(inputs)
            new_mask = new_mask.squeeze().numpy()
            pred = pred.cpu().squeeze().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            end = time.time()
            dc = Dice(pred, new_mask)
            id_list.append(ID[0])
            dc_list.append(dc)
            dataset_list.append('val')
            category_list.append(classes[i])
            time_list.append(end - begin)
            print(f"{ID[0]}\t{classes[i]}\t{dc * 100:.2f}%")

    for _, (ID, img_path, classes) in enumerate(test_loader):
        patches = test_set.generate_inf_patch(img_path[0])
        patch_loader = DataLoader(patches, batch_size)
        classes = torch.cat(classes)
        classes = classes.tolist()
        for i, (new_img, new_mask, landmark) in enumerate(patch_loader):
            new_img = new_img.to(device)
            heatmap = gaussian(landmark.to(device))
            inputs = torch.cat([new_img, heatmap], dim=1)
            begin = time.time()
            with torch.no_grad():
                pred = model(inputs)
            new_mask = new_mask.squeeze().numpy()
            pred = pred.cpu().squeeze().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            end = time.time()
            dc = Dice(pred, new_mask)
            id_list.append(ID[0])
            dc_list.append(dc)
            dataset_list.append('val')
            category_list.append(classes[i])
            time_list.append(end - begin)
            print(f"{ID[0]}\t{classes[i]}\t{dc * 100:.2f}%")

    df = pd.DataFrame()
    df['Dice'] = dc_list
    df['Category'] = category_list
    df['ID'] = id_list
    df['Set'] = dataset_list
    df['Time'] = time_list
    df.to_csv("result/test_segmentation_mse.csv")
