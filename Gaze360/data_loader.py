import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import cv2

trans_train = transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),	# this also convert pixel value from [0,255] to [0,1]
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
														 std=[0.229, 0.224, 0.225]),
		])

trans = transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),	# this also convert pixel value from [0,255] to [0,1]
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
														 std=[0.229, 0.224, 0.225]),
		])


def get_train_loader(data_dir, batch_size, load_mode, num_workers=4, is_shuffle=True):

	# load dataset
	refer_list_file = 'train_test_split.json'
	print('load the train file list from: ', refer_list_file)

	with open(refer_list_file, 'r') as f:
		datastore = json.load(f)

	sub_folder_use = 'train'
	data_dir = os.path.join(data_dir, sub_folder_use)
	train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
				transform=trans, is_shuffle=is_shuffle, is_load_label=True, load_mode=load_mode)
	train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

	return train_loader


def get_test_loader(data_dir, batch_size, load_mode, num_workers=4, is_shuffle=True):

	# load dataset
	refer_list_file = 'train_test_split.json'
	print('load the train file list from: ', refer_list_file)

	with open(refer_list_file, 'r') as f:
		datastore = json.load(f)

	sub_folder_use = 'test'
	data_dir = os.path.join(data_dir, sub_folder_use)
	test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
				transform=trans, is_shuffle=is_shuffle, is_load_label=True, load_mode=load_mode)
	test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

	return test_loader

def get_validation_loader(data_dir, batch_size, num_workers=4, is_shuffle=True):

	# load dataset
	refer_list_file = 'train_test_split.json'
	print('load the train file list from: ', refer_list_file)

	with open(refer_list_file, 'r') as f:
		datastore = json.load(f)

	sub_folder_use = 'val'
	data_dir = os.path.join(data_dir, sub_folder_use)
	val_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
				transform=trans, is_shuffle=is_shuffle, is_load_label=True, load_mode=load_mode)
	val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

	return val_loader

class GazeDataset(Dataset):
	def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
			index_file=None, is_load_label=True, load_mode = ''):
		self.path = dataset_path
		self.hdfs = {}
		self.sub_folder = sub_folder
		self.is_load_label = is_load_label
		self.load_mode = load_mode
		
		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0

		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i])
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			assert self.hdfs[num_i].swmr_mode

		# Construct mapping from full-data index to key and person-specific index
		if index_file is None:
			self.idx_to_kv = []
			
			for num_i in range(0, len(self.selected_keys)):
				n = self.hdfs[num_i]["face"].shape[0]
				self.idx_to_kv += [(num_i, i) for i in range(n)]
		else:
			print('load the file: ', index_file)
			self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		if is_shuffle:
			random.shuffle(self.idx_to_kv)  # random the order to stable the training

		self.hdf = None
		self.transform = transform

	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

	def __getitem__(self, idx):
		key, idx = self.idx_to_kv[idx]

		self.hdf = h5py.File(os.path.join(self.path, self.selected_keys[key]), 'r', swmr=True)
		assert self.hdf.swmr_mode

		# Get face image
		image = self.hdf['face'][idx, :]
		image = image[:, :, [2, 1, 0]]  # from BGR to RGB
		image = image.astype(np.uint8)
		image = self.transform(image)

		# Get eye images
		left = self.hdf['left'][idx, :]
		left = left[:, :, [2, 1, 0]]
		left = left.astype(np.uint8)
		left = self.transform(left)

		right = self.hdf['right'][idx, :]
		right = right[:, :, [2, 1, 0]]
		right = right.astype(np.uint8)
		right = self.transform(right)

		if self.load_mode == 'load_single_face':
			images = {"face": image}
		elif self.load_mode == 'load_multi_region':
			images = {"face": image, "left_eye": left, "right_eye": right}


		# Get labels
		if self.is_load_label:
			gaze_label = self.hdf['gaze'][idx, :]
			gaze_label = gaze_label.astype('float')
			return images, gaze_label
		else:
			return images
			
