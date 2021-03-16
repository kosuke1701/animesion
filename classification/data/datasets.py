import os
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

class ZACI20(data.Dataset):
	def __init__(self, dataset_dir, input_size=224, split='train', transform=None, seed=1234):
		self.split = split
		if self.split=='train':
			if transform is None:
				transform = transforms.Compose([
					transforms.Resize((input_size+32, input_size+32)),
					transforms.RandomCrop((input_size, input_size)),
					transforms.RandomHorizontalFlip(),
					transforms.ColorJitter(brightness=0.1, 
					contrast=0.1, saturation=0.1, hue=0.1),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5, 0.5, 0.5],
										std=[0.5, 0.5, 0.5])
				])
		elif self.split=='val' or self.split=='test':
			if transform is None:
				transform = transforms.Compose([
					transforms.Resize((input_size, input_size)), 
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])	

		self.base = ImageFolder(dataset_dir, transform=transform)
		self.no_classes = len(self.base.classes)
		self.classes = self.base.classes

		index = list(range(len(self.base)))
		train_index, dev_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=seed)
		dev_index, test_index = train_test_split(dev_index, test_size=0.5, shuffle=False)

		if self.split == "train":
			self.base = Subset(self.base, train_index)
		elif self.split == "val":
			self.base = Subset(self.base, dev_index)
		elif self.split == "test":
			self.base = Subset(self.base, test_index)
			
		
	
	def __getitem__(self, idx):
		return self.base[idx]
	
	def __len__(self):
		return len(self.base)

class moeImouto(data.Dataset):
	'''
	https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home
	http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
	https://github.com/nagadomi/lbpcascade_animeface
	'''
	def __init__(self, root, input_size=224, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size+32, self.input_size+32)),
				transforms.RandomCrop((self.input_size, self.input_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1, 
				contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])
			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size, self.input_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.no_classes = len(self.classes)
		

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)

class danbooruFaces(data.Dataset):
	'''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
	def __init__(self, root, input_size=224, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size+32, self.input_size+32)),
				transforms.RandomCrop((self.input_size, self.input_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1, 
				contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])
		elif self.split=='val':
			print('Validation set')
			self.set_dir = os.path.join(self.root, 'val.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size, self.input_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])	
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size, self.input_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.no_classes = len(self.classes)
		
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)

class cartoonFace(data.Dataset):
	'''
	http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d
	https://github.com/luxiangju-PersonAI/iCartoonFace
	'''
	def __init__(self, root, input_size=128, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size+32, self.input_size+32)),
				transforms.RandomCrop((self.input_size, self.input_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1, 
				contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])
			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
			if self.transform is None:
				self.transform = transforms.Compose([
				transforms.Resize((self.input_size, self.input_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.no_classes = len(self.classes)
		

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)