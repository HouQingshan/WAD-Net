import copy
import random
import os
import shutil
import albumentations as A
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.models as models
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image, ImageCms
from collections import OrderedDict, defaultdict
from urllib.request import urlretrieve
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from WAD_Net.networks.MIL50_model import Create_MIL_model

cudnn.benchmark = True
torch.autograd.set_detect_anomaly = True

image_dir = '/home/houqs/EyeIQA/EyeQ_baseline/data/train'
label_dir = './train_Labels.csv'
#image_dir = '/home/houqs/EyeIQA/EyeQ_baseline/data/train'
#label_dir = './new_data_label.csv'
label_dir = pd.read_csv(label_dir)

image_dir_val = '/home/houqs/EyeIQA/EyeQ_baseline/data/val'
label_dir_val = './val_Labels.csv'
#image_dir_val = '/home/houqs/EyeIQA/EyeQ_baseline/data/train'
#label_dir_val = './new_data_label_val.csv'
label_dir_val = pd.read_csv(label_dir_val)


class DatasetGenerator(Dataset):
	def __init__(self, images_filepaths, label_filepaths, total=None, transform=None):
		self.images_filepaths = images_filepaths
		self.label_filepaths = label_filepaths
		if (total is not None):
			self.label_filepaths = self.label_filepaths[:total]
		self.transform = transform
		
	def __len__(self):
		return len(self.label_filepaths)

	def __getitem__(self, idx):
		#image_filepath = os.path.join(self.images_filepaths, self.label_filepaths.iloc[idx].image)
		image_filepath = os.path.join(self.images_filepaths, self.label_filepaths.iloc[idx].image+'.png')
		# print(image_filepath)
		image = cv2.imread(image_filepath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform is not None:
			image = self.transform(image=image)["image"]
		
		#label = torch.tensor(self.label_filepaths.iloc[idx].DR_grade)
		label = torch.tensor(self.label_filepaths.iloc[idx].level)
		return image, label, image_filepath

train_transform = A.Compose(
	[
		A.SmallestMaxSize(max_size=800),
		A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
		A.RandomCrop(height=768, width=768),
		A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
		A.RandomBrightnessContrast(p=0.5),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	]
)


train_dataset = DatasetGenerator(images_filepaths=image_dir, label_filepaths=label_dir, transform=train_transform)

val_transform = A.Compose(
	[
		A.SmallestMaxSize(max_size=800),
		A.CenterCrop(height=768, width=768),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	]
)

val_dataset = DatasetGenerator(images_filepaths=image_dir_val, label_filepaths=label_dir_val, transform=val_transform)

###
def calculate_accuracy(output, target):
	output = torch.softmax(output, dim=1)
	_, predictions = output.max(dim=1)
	Accuracy = torch.true_divide((target == predictions).sum(dim=0), output.size(0)).item()
	return Accuracy



class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision  #
		self.reset()
	
	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
	
	def update(self, metric_name, val):
		metric = self.metrics[metric_name]
		
		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]
		return metric["avg"]
	
	#
	def __str__(self):
		return " | ".join(
			[
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)


# Define training parameters
params = {
	"model": "resnet50",
	"device": "cuda",
	"lr": 0.0001,
	"batch_size": 10,
	"num_workers": 6,
	"epochs": 150,
}

# params = {
# 	"model": "resnet50",
# 	"device": "cuda",
# 	"lr": 0.001,
# 	"batch_size": 10,
# 	"num_workers": 6,
# 	"epochs": 150,
# }

### 创建并载入病灶判别网络权重
# log_lesion_dir = "./PTH/resnet50_for_2.pth"
# model_lesion = getattr(models, params["model"])(pretrained=False, num_classes=1)
# if os.path.exists(log_lesion_dir):
# 	checkpoint = torch.load(log_lesion_dir)
# 	model_lesion.load_state_dict(checkpoint['model'])
# 	print('病灶判别网络权重载入成功！')
# else:
# 	print('病灶判别网络权重载入失败')
	
log_lesion_dir = "./PTH/resnet50_for_2_fake.pth"
model_lesion = getattr(models, params["model"])(pretrained=False, num_classes=1)
if os.path.exists(log_lesion_dir):
	checkpoint = torch.load(log_lesion_dir)
	updata_state_dict = OrderedDict()
	for k, v in checkpoint['model'].items():
		k_new = k[7:]
		updata_state_dict[k_new] = v
	model_lesion.load_state_dict(updata_state_dict)
	print('病灶判别网络权重载入成功！')
else:
	print('病灶判别网络权重载入失败')
	
###########################
model_cls = Create_MIL_model()
model_cls_dict = model_cls.state_dict()

model_pre = getattr(models, params["model"])(pretrained=True)
model_pre_dict = model_pre.state_dict()

new_state_dict = OrderedDict()
for k, v in model_pre_dict.items():
	if k != 'fc.weight' and k != 'fc.bias':
		new_state_dict[k] = v
		
model_cls_dict.update(new_state_dict)
model_cls.load_state_dict(model_cls_dict)
######################################################

from torch.nn import DataParallel
model_lesion = DataParallel(model_lesion, device_ids=[0, 1])
model_cls = DataParallel(model_cls, device_ids=[0, 1])

###########################
model_lesion = model_lesion.to(params["device"])
model_cls = model_cls.to(params["device"])

criterion = nn.CrossEntropyLoss().to(params["device"])
optimizer = torch.optim.Adam(model_cls.parameters(), lr=params["lr"])

train_loader = DataLoader(
	train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True
)
val_loader = DataLoader(
	val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True
)


def train(train_loader, model_lesion, model_cls, criterion, optimizer, epoch, params):
	metric_monitor = MetricMonitor()
	model_lesion.eval()
	model_cls.train()
	stream = tqdm(train_loader)
	for i, (images, target, image_label_filepath) in enumerate(stream, start=1):
		
		image = images.to(params["device"], non_blocking=True)
		target = target.to(params["device"], non_blocking=True)
		images_patch = rearrange(image, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=128, p2=128)
		output_patch_res = model_lesion(images_patch)
		images_patch_copy = images_patch.clone()
		#### softmax操作 images patch化以及逆操作
		score = torch.sigmoid(output_patch_res)
		for i in range(len(score)):
			if score[i] == 0:
				pass
			else:
				images_patch_copy[i,:,:,:] = images_patch[i,:,:,:] * score[i]
				
		images_score = rearrange(images_patch_copy, '(b h w) c p1 p2 -> b c (h p1) (w p2)', b=image.shape[0], h=6)
		output_cls = model_cls(images_score)
		loss = criterion(output_cls, target)
		
		accuracy = calculate_accuracy(output_cls, target)
		loss2 = metric_monitor.update("Loss", loss.item())
		Accuracy2 = metric_monitor.update("Accuracy", accuracy)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		stream.set_description(
			"Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
		)
	return loss2, Accuracy2

def validate(val_loader, model_lesion, model_cls,  criterion, epoch, params):
	metric_monitor = MetricMonitor()
	model_lesion.eval()
	model_cls.eval()
	stream = tqdm(val_loader)
	with torch.no_grad():
		for i, (images, target, image_label_filepath) in enumerate(stream, start=1):
			
			image = images.to(params["device"], non_blocking=True)
			target = target.to(params["device"], non_blocking=True)
			images_patch = rearrange(image, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=128, p2=128)
			output_patch_res = model_lesion(images_patch)
			images_patch_copy = images_patch.clone()
			#### softmax操作 images patch化以及逆操作
			score = torch.sigmoid(output_patch_res)
			for i in range(len(score)):
				if score[i] == 0:
					pass
				else:
					images_patch_copy[i, :, :, :] = images_patch[i, :, :, :] * score[i]
			
			images_score = rearrange(images_patch_copy, '(b h w) c p1 p2 -> b c (h p1) (w p2)', b=image.shape[0],
			                         h=6)
			output_cls = model_cls(images_score)
			loss = criterion(output_cls, target)
			accuracy = calculate_accuracy(output_cls, target)
			
			loss1 = metric_monitor.update("Loss", loss.item())
			accuracy1 = metric_monitor.update("Accuracy", accuracy)
			
			stream.set_description(
				"Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
			)
	
	return loss1, accuracy1


log_MIL_dir = "./PTH_MIL/MIL50_train.pth"

if os.path.exists(log_MIL_dir):
	checkpoint = torch.load(log_MIL_dir)
	model_cls.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	start_epoch = checkpoint['epoch']
	print('loading epoch {} success！'.format(start_epoch))
else:
	start_epoch = 0
	print('无MIL模型权重保存，将从头开始训练！')

best_metric = 0
for epoch in range(start_epoch + 1, start_epoch + 1 + params["epochs"]):
	loss_val2, accuracy_val2 = train(train_loader, model_lesion, model_cls, criterion, optimizer, epoch, params)
	loss_val, accuracy_val = validate(val_loader, model_lesion, model_cls, criterion, epoch, params)
	if best_metric < accuracy_val2:
		best_metric = accuracy_val2
		logdir = "./PTH_MIL/MIL50_train.pth"
		state = {'model': model_cls.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
		torch.save(state, logdir)
		print('save epoch {} success！'.format(epoch))
