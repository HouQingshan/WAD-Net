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




image_dir = '/home/houqs/EyeIQA/EyeQ_baseline/data/test'
label_dir = './test_Labels.csv'

label_dir = pd.read_csv(label_dir)


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


test_transform = A.Compose(
	[
		A.SmallestMaxSize(max_size=800),
		A.CenterCrop(height=768, width=768),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	]
)

test_dataset = DatasetGenerator(images_filepaths=image_dir, label_filepaths=label_dir, transform=test_transform)


###
def calculate_accuracy(output, target):
	output = torch.softmax(output, dim=1)
	_, predictions = output.max(dim=1)
	Accuracy = torch.true_divide((target == predictions).sum(dim=0), output.size(0)).item()
	return Accuracy


###
# def Cohen_kappa_score(output, target):
# 	output = torch.softmax(output, dim=1).squeeze().cpu().detach()
# 	_, predictions = output.max(dim=1)
# 	target = target.squeeze().cpu().detach()
# 	# predictions2 = predictions+1
# 	# target2 = target+1
# 	kappa = cohen_kappa_score(predictions, target ,weights='quadratic')
# 	return kappa

outputlist = []
targetlist = []
def Cohen_kappa_score1(output, target):
	output = torch.softmax(output, dim=1).squeeze().cpu().detach()
	_, predictions = output.max(dim=1)
	target = target.squeeze().cpu().detach()
	predictions_list = predictions.tolist()
	target_list = target.tolist()
	outputlist.extend(predictions_list)
	targetlist.extend(target_list)
	return outputlist, targetlist

class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision  # 精度设置3位小数
		self.reset()
	
	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
	
	def update(self, metric_name, val):
		metric = self.metrics[metric_name]
		
		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]
		return metric["avg"]
	
	# 当使用print输出对象的时候，只要自己定义了__str__(self)
	# 方法，那么就会打印从在这个方法中return的数据
	def __str__(self):
		return " | ".join(
			[
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)
	
params = {
	"model": "resnet50",
	"device": "cuda",
	"batch_size": 10,
	"num_workers": 6,
}

log_lesion_dir = "./PTH/resnet50_for_2_fake.pth"
model_lesion = getattr(models, params["model"])(pretrained=False, num_classes=1)
if os.path.exists(log_lesion_dir):
	checkpoint = torch.load(log_lesion_dir)
	updata_state_dict = OrderedDict()
	for k, v in checkpoint['model'].items():
		k_new = k[7:]
		updata_state_dict[k_new] = v
	model_lesion.load_state_dict(updata_state_dict)
	print('病灶判别网络权重载入成功!')
else:
	print('病灶判别网络权重载入失败!')

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

from torch.nn import DataParallel
model_lesion = DataParallel(model_lesion, device_ids=[0,1])
model_cls = DataParallel(model_cls, device_ids=[0,1])

model_lesion = model_lesion.to(params["device"])
model_cls = model_cls.to(params["device"])

test_loader = DataLoader(
	test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True
)

def test(test_loader, model_lesion, model_cls,  params):
	metric_monitor = MetricMonitor()
	model_lesion.eval()
	model_cls.eval()
	stream = tqdm(test_loader)
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
			accuracy = calculate_accuracy(output_cls, target)
			outputlist, targetlist = Cohen_kappa_score1(output_cls, target)
			peri = torch.tensor(outputlist).squeeze()
			tar = torch.tensor(targetlist).squeeze()
			kappa = cohen_kappa_score(peri, tar, weights='quadratic')
			# kappa = Cohen_kappa_score(output_cls, target)
			
			accuracy1 = metric_monitor.update("Accuracy", accuracy)
			#kappa1 = metric_monitor.update("kappa", kappa)
			print("kappa:", kappa)
			stream.set_description(
				"test. {metric_monitor}".format(metric_monitor=metric_monitor)
			)
	
	return accuracy1, kappa

log_MIL_dir = "./PTH_MIL/MIL50_train.pth"

if os.path.exists(log_MIL_dir):
	checkpoint = torch.load(log_MIL_dir)
	#updata_state_dict = OrderedDict()
	#for k, v in checkpoint['model'].items():
	#	k_new = k[7:]
	#	updata_state_dict[k_new] = v
	#model_cls.load_state_dict(updata_state_dict)
	model_cls.load_state_dict(checkpoint['model'])
	print('MIL网络权重载入成功！')
else:
	print('MIL网络权重载入失败!')

	
accuracy1, kappa1 = test(test_loader, model_lesion, model_cls, params)
print(f"accuracy1:{accuracy1}, kappa1:{kappa1}")