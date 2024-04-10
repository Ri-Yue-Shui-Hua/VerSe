# -*- coding : UTF-8 -*-
# @file   : train_seg.py
# @Time   : 2024-03-28 17:01
# @Author : wmz
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


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def train_seg(args):
	device = args.device if torch.cuda.is_available() else 'cpu'
	epochs = args.epochs
	batch_size = args.batch_size
	number_class = args.num_class
	eval_epoch = args.eval_epoch
	log_inter = args.log_inter
	base_lr = args.lr
	root_dir = args.root
	image_suffix = args.img_suffix
	save_name = str(batch_size)
	save_path = args.log_path
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
	start_epoch = 0
	model = UNet(2, 1).to(device)
	loss_func = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=base_lr)
	scheduler = MultiStepLR(optimizer, milestones=[epochs//3, epochs//3*2], gamma=0.1)
	gaussian = Gaussian(3, None, 5, norm=True).to(device)
	for ep in tqdm(range(start_epoch, epochs)):
		tqdm.write(f"----------{ep}----------")
		model.train()
		train_loss = 0.
		train_dc = 0.
		training_all_loss = 0.0
		acc = 0
		begin = time.time()
		for _, (ID, img_path) in enumerate(train_loader):
			patches = train_set.generate_train_patch(img_path[0])
			patch_loader = DataLoader(patches, 1)
			for i, (image, mask, landmark) in enumerate(patch_loader):
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
		train_loss /= NUM_BATCHES
		train_dc /= NUM_BATCHES
		logger.info(
					f"Epoch:{ep + 1}/{epochs}\ttrain_loss:{train_loss:.6f}\ttrain_dice:{train_dc * 100:.2f}%\tTime:{end - begin:.3f}s"
				)


		for i, (data, target) in enumerate(train_set):
			optimizer.zero_grad()
			data = data.to(device)
			target = target.to(device)
			output = model(data)
			loss = loss_func(output, target)
			loss.backward()
			optimizer.step()
			training_all_loss += loss.item()
			tqdm.write(
				f"[*]train finish, all loss is: {training_all_loss / (i + 1):8f}")
			writer.add_scalar('Training All Loss', training_all_loss / (i + 1), epoch)
		scheduler.step()

		if epoch % eval_epoch == eval_epoch - 1:
			model.eval()
			testing_all_loss = 0
			acc = 0
			with torch.no_grad():
				for i, (data, target) in enumerate(test_set):
					data = data.to(device)
					target = target.to(device)
					output = model(data)
					loss = loss_func(output, target)
					testing_all_loss += loss.item()
				tqdm.write(f"[*]eval on test finish, loss is: {testing_all_loss/(i+1):.8f}%")
				writer.add_scalar('Eval on Test All Loss', testing_all_loss / (i + 1), epoch)
				checkpoint = {
					"net": model.state_dict(),
					"epoch": epoch,
				}
				torch.save(checkpoint, save_path + save_name + "_" + str(epoch) + ".pth")


if __name__ == "__main__":
	setup_seed(33)
	parser = argparse.ArgumentParser(description='Segmentation training')
	parser.add_argument("--device", type=str, default='cuda')
	parser.add_argument('-lr', default=0.0001, type=float, help='learning rate')
	parser.add_argument('-batch_size', default=1, type=int, help='batch size')
	parser.add_argument('-epochs', default=100, type=int, help='training epochs')
	parser.add_argument('-eval_epoch', default=1, type=int, help='evaluation epoch')
	parser.add_argument('-log_path', default="logs", type=str, help='the path of the log')
	parser.add_argument('-log_inter', default=50, type=int, help='log interval')
	parser.add_argument('-read_params', default=False, type=bool, help='if read pretrained params')
	parser.add_argument('-params_path', default="", type=str, help='the path of the pretrained model')
	parser.add_argument('-basepath', default="", type=str, help='base dataset path')
	parser.add_argument('-augmentation', default=False, type=bool, help='if augmentation')
	parser.add_argument('-num_class', default=25, type=int, help='the number of class')
	parser.add_argument('-model_name', default="", type=str, help='ResUnet')

	args = parser.parse_args()


