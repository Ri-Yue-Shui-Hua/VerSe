# -*- coding : UTF-8 -*-
# @file   : CommonDataSet.py
# @Time   : 2024-03-28 18:22
# @Author : wmz

import json
import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


def get_json(ctd_path):
	with open(ctd_path) as json_data:
		dict_list = json.load(json_data)
		json_data.close()
	ctd_list = []
	for d in dict_list:
		if 'direction' in d:
			ctd_list.append(tuple(d['direction']))
		elif 'nan' in str(d):
			continue
		else:
			ctd_list.append(d['label'], d['X'], d['Y'], d['Z'])
	return ctd_list


if __name__ == "__main__":
	root_path = r""




