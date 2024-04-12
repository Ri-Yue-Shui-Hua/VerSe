# -*- coding : UTF-8 -*-
# @file   : CommonDataSet.py
# @Time   : 2024-03-28 18:22
# @Author : wmz

import json
import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Tuple
import random
from batchgenerators.transforms.color_transforms import *
from batchgenerators.transforms.spatial_transforms import *
from batchgenerators.transforms.crop_and_pad_transforms import *
from batchgenerators.transforms.utility_transforms import *
from batchgenerators.transforms.sample_normalization_transforms import *
from batchgenerators.transforms.noise_transforms import *
from batchgenerators.transforms.resample_transforms import *
from batchgenerators.transforms.abstract_transforms import Compose


def get_json_label_coord(point_file):
	with open(point_file, "r") as f:
		info = json.load(f)
		f.close()
	direction_str = info[0]["direction"]
	label_num = len(info)
	label_list = []
	ijk_coord_list = []  # 在lps坐标系下的jik坐标
	for idx in range(1, label_num):
		label = info[idx]["label"]
		coord = [info[idx]['X'], info[idx]['Y'], info[idx]['Z']]
		label_list.append(int(label))
		ijk_coord_list.append(coord)
	# print("label:", label, "coord: ", coord)
	return label_list, ijk_coord_list


def resampleCropImage(image, outspacing, lps_center, radius, direction, interpolateMethod=sitk.sitkLinear):
	"""
	将体数据重采样的指定的spacing大小\n
	paras：
	outpacing：指定的spacing，例如[1,1,1]
	vol：sitk读取的image信息，这里是体数据\n
	return：重采样后的数据
	"""
	outsize = [0, 0, 0]
	default_value = np.float64(sitk.GetArrayViewFromImage(image).min())
	# 读取文件的size和spacing信息
	transform = sitk.Transform()
	transform.SetIdentity()
	# 计算改变spacing后的size，用物理尺寸/体素的大小
	outsize[0] = round(2 * radius[0])
	outsize[1] = round(2 * radius[1])
	outsize[2] = round(2 * radius[2])
	inOrigin = image.GetOrigin()
	outOrigin = list(inOrigin)
	# outOrigin[0] = lps_center[0] - radius[0]
	# outOrigin[1] = lps_center[1] - radius[1]
	# outOrigin[2] = lps_center[2] - radius[2]
	outOrigin = lps_center - np.matmul(direction, np.array(radius) * outspacing)

	# 设定重采样的一些参数
	resampler = sitk.ResampleImageFilter()
	resampler.SetTransform(transform)
	resampler.SetInterpolator(interpolateMethod)
	resampler.SetOutputOrigin(outOrigin)
	resampler.SetOutputSpacing(outspacing)
	resampler.SetDefaultPixelValue(default_value)
	resampler.SetOutputDirection(image.GetDirection())
	resampler.SetSize(outsize)
	newImage = resampler.Execute(image)
	return newImage


def save_image_label_from_array(img_arr, category, origin, spacing, direction, label_flag=False):
	itk_img = sitk.GetImageFromArray(img_arr)
	itk_img.SetOrigin(origin)
	itk_img.SetSpacing(spacing)
	itk_img.SetDirection(direction)
	if label_flag:
		sitk.WriteImage(itk_img, f"./DATA/{category}_label.nii.gz")
	else:
		sitk.WriteImage(itk_img, f"./DATA/{category}.nii.gz")


def save_image_label_from_itk_img(image, category, label_flag=False):
	if label_flag:
		sitk.WriteImage(image, f"./DATA/{category}_label.nii.gz")
	else:
		sitk.WriteImage(image, f"./DATA/{category}.nii.gz")


def choose_T(T):
	idx = np.random.randint(3)
	t = []
	t.append(Compose(T))
	t.append(T[np.random.randint(len(T))])
	t.append(None)
	print(idx)
	return t[idx]


def transform(d, gt):
	T = []
	single_channel_size = d.shape[1:]
	d = d[np.newaxis, :, :, :, :]
	gt = gt[np.newaxis, :, :, :, :]

	T.append(GaussianNoiseTransform(p_per_sample=0.5))
	T.append(GaussianBlurTransform((0.5, 3), different_sigma_per_channel=False, p_per_sample=0.8))
	T.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), per_channel=False, p_per_sample=0.5))
	T.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.5))
	T.append(GammaTransform(gamma_range=(0.7, 1.5), retain_stats=True, p_per_sample=0.5))
	axis = [2]
	T.append(MirrorTransform(data_key='data', label_key='gt', axes=axis))

	T = choose_T(T)
	if T is not None:
		out_dict = T(data=d, gt=gt)
		d, gt = out_dict.get('data'), out_dict.get('gt')
	d = d[0]
	gt = gt[0]
	return d, gt


class CommonDataset(Dataset):
	def __init__(
			self,
			root_dir,
			file_list: List[str],
			mode: str = 'train',
			patch_size: Tuple[int] = (128, 128, 96)
	):
		self.root_dir = root_dir
		self.file_list = file_list
		self.mode = mode
		self.patch_size = patch_size

	def generate_train_patch(self, img_file):
		img_list = []
		mask_list = []
		patch_size = list(self.patch_size)
		point_file = img_file.replace("_ct.nii.gz", "_seg-subreg_ctd.json")
		label_list, coord_list = get_json_label_coord(point_file)
		mask_file = img_file.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
		mask = sitk.ReadImage(mask_file)
		image = sitk.ReadImage(img_file)
		origin = image.GetOrigin()
		spacing = image.GetSpacing()
		direction = image.GetDirection()

		for idx, c in enumerate(label_list):
			ijk_coord = list(coord_list[idx])
			lps_direction = np.array(list(direction)).reshape(3, 3)
			lps_center = np.matmul(lps_direction, np.array(ijk_coord) * spacing)
			lps_center = origin + lps_center
			# print("lps_center: ", lps_center)
			radius = np.array(patch_size) / 2
			new_img = resampleCropImage(image, outspacing=spacing, lps_center=lps_center, radius=radius,
			                            direction=lps_direction)
			new_mask = resampleCropImage(mask, outspacing=spacing, lps_center=lps_center, radius=radius,
			                             direction=lps_direction, interpolateMethod=sitk.sitkNearestNeighbor)
			# save_image_label_from_itk_img(new_img, c)
			# save_image_label_from_itk_img(new_mask, c, label_flag=True)
			img_arr = sitk.GetArrayFromImage(new_img).astype(np.float32)
			mask_arr = sitk.GetArrayFromImage(new_mask).astype(np.float32)
			mask_arr[mask_arr != c] = 0
			mask_arr[mask_arr == c] = 1
			# save_image_label_from_array(img_arr, c, origin, spacing, direction, label_flag=False)
			# save_image_label_from_array(mask_arr, c, origin, spacing, direction, label_flag=True)
			img_arr = np.expand_dims(img_arr, 0)
			mask_arr = np.expand_dims(mask_arr, 0)
			img_arr, mask_arr = transform(img_arr, mask_arr)
			img_list.append(img_arr)
			mask_list.append(mask_arr)
		new_img = torch.Tensor(np.vstack(img_list)).unsqueeze(1)
		new_img = torch.clip(new_img / 2048, -1, 1)
		new_mask = torch.Tensor(np.vstack(mask_list)).unsqueeze(1)
		landmark = torch.zeros_like(new_mask, dtype=torch.float32)
		patch_size = list(self.patch_size)[::-1]
		landmark[:, 0, patch_size[0] // 2,
		patch_size[1] // 2, patch_size[2] // 2] = 1
		# 添加数据增强
		return TensorDataset(new_img, new_mask, landmark)

	def generate_random_patch(self, image_path):
		image = sitk.ReadImage(image_path)
		mask_path = image_path.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
		mask = sitk.ReadImage(mask_path)
		point_file = image_path.replace("_ct.nii.gz", "_seg-subreg_ctd.json")
		class_list, coord_list = get_json_label_coord(point_file)
		idx = random.randint(0, len(class_list))
		label = class_list[idx]
		ijk_coord = coord_list[idx]
		patch_size = list(self.patch_size)
		origin = image.GetOrigin()
		spacing = image.GetSpacing()
		direction = image.GetDirection()
		direction = np.array(list(direction)).reshape(3, 3)
		lps_center = np.matmul(direction, np.array(ijk_coord) * spacing)
		lps_center = origin + lps_center
		print("lps_center: ", lps_center)
		radius = np.array(patch_size) / 2
		new_img = resampleCropImage(image, outspacing=spacing, lps_center=lps_center, radius=radius,
		                            direction=direction)
		new_mask = resampleCropImage(mask, outspacing=spacing, lps_center=lps_center, radius=radius,
		                             direction=direction,
		                             interpolateMethod=sitk.sitkNearestNeighbor)
		# sitk.WriteImage(new_img, f"./DATA/{label}.nii.gz")
		# sitk.WriteImage(new_mask, f"./DATA/{label}_label.nii.gz")
		img_arr = sitk.GetArrayFromImage(new_img).astype(np.float32)
		mask_arr = sitk.GetArrayFromImage(new_mask).astype(np.float32)
		mask_arr[mask_arr != label] = 0
		mask_arr[mask_arr == label] = 1
		patch_size = list(self.patch_size)[::-1]
		landmark = np.zeros(patch_size, dtype=np.float32)
		landmark[tuple([shape // 2 for shape in patch_size])] = 1
		return img_arr, mask_arr, landmark, label

	def generate_inf_patch(self, img_file):
		img_list = []
		mask_list = []
		patch_size = list(self.patch_size)
		point_file = img_file.replace("_ct.nii.gz", "_seg-subreg_ctd.json")
		label_list, coord_list = get_json_label_coord(point_file)
		mask_file = img_file.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
		mask = sitk.ReadImage(mask_file)
		image = sitk.ReadImage(img_file)
		origin = image.GetOrigin()
		spacing = image.GetSpacing()
		direction = image.GetDirection()

		for idx, c in enumerate(label_list):
			ijk_coord = list(coord_list[idx])
			direction = np.array(list(direction)).reshape(3, 3)
			lps_center = np.matmul(direction, np.array(ijk_coord) * spacing)
			lps_center = origin + lps_center
			# print("lps_center: ", lps_center)
			radius = np.array(patch_size) / 2
			new_img = resampleCropImage(image, outspacing=spacing, lps_center=lps_center, radius=radius,
			                            direction=direction)
			new_mask = resampleCropImage(mask, outspacing=spacing, lps_center=lps_center, radius=radius,
			                             direction=direction,
			                             interpolateMethod=sitk.sitkNearestNeighbor)
			# sitk.WriteImage(new_img, f"./DATA/{c}.nii.gz")
			# sitk.WriteImage(new_mask, f"./DATA/{c}_label.nii.gz")
			img_arr = sitk.GetArrayFromImage(new_img).astype(np.float32)
			mask_arr = sitk.GetArrayFromImage(new_mask).astype(np.float32)
			mask_arr[mask_arr != c] = 0
			mask_arr[mask_arr == c] = 1

			img_list.append(np.expand_dims(img_arr, 0))
			mask_list.append(np.expand_dims(mask_arr, 0))
		new_img = torch.Tensor(np.vstack(img_list)).unsqueeze(1)
		new_img = torch.clip(new_img / 2048, -1, 1)
		new_mask = torch.Tensor(np.vstack(mask_list)).unsqueeze(1)
		landmark = torch.zeros_like(new_mask, dtype=torch.float32)
		patch_size = list(self.patch_size)[::-1]
		landmark[:, 0, patch_size[0] // 2,
		patch_size[1] // 2, patch_size[2] // 2] = 1
		return TensorDataset(new_img, new_mask, landmark)

	def _get_inf_data(self, index):
		path = self.file_list[index]
		basename = os.path.basename(path)
		ID = basename[:basename.find("_ct.nii.gz")]
		img_path = os.path.join(self.root_dir, path)
		point_file = img_path.replace("_ct.nii.gz", "_seg-subreg_ctd.json")
		label_list, coord_list = get_json_label_coord(point_file)
		return ID, img_path, label_list

	def _get_train_data(self, index):
		path = self.file_list[index]
		basename = os.path.basename(path)
		ID = basename[:basename.find("_ct.nii.gz")]
		img_path = os.path.join(self.root_dir, basename)
		image, mask, landmark, category = self.generate_random_patch(img_path)
		image = normalize(image.astype(np.float32))
		image = np.expand_dims(image, 0)
		mask = np.expand_dims(mask, 0)
		landmark = np.expand_dims(landmark, 0)
		return ID, image, mask, landmark, category

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, item):
		image_file = os.path.join(self.root_dir, self.file_list[item])
		if self.mode == 'train':
			return self._get_inf_data(item)
		# return self._get_train_data(item)
		return self._get_inf_data(item)


def normalize(img: np.ndarray):
	'''Intensity value of the CT volumes is divided by 2048 and clamped between -1 and 1'''
	return np.clip(img / 2048, -1, 1)


def get_files(path, suffix):
	return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]


if __name__ == "__main__":
	root_dir = r"E:\Data\VerSe\test\1mm"
	image_suffix = "_ct.nii.gz"
	mode = "train"
	image_list = get_files(root_dir, image_suffix)
	common_dataset = CommonDataset(root_dir, image_list, mode)
	dataloader = DataLoader(common_dataset)
	# for test
	# for _, (ID, img_path) in enumerate(dataloader):
	# 	patches = common_dataset.generate_inf_patch(img_path[0])
	# 	patch_loader = DataLoader(patches, 1)
	# 	for i, (new_img, new_mask, landmark) in enumerate(patch_loader):
	# 		new_img = new_img
	# 		exit(0)
	# for train
	# for _, (ID, image, mask, landmark, category) in enumerate(dataloader):
	# 	print("ID: ", ID)

	for _, (ID, img_path) in enumerate(dataloader):
		patches = common_dataset.generate_train_patch(img_path[0])
		patch_loader = DataLoader(patches, 1)
		for i, (new_img, new_mask, landmark) in enumerate(patch_loader):
			print("i: ", i)
			new_img = new_img
		exit(0)
