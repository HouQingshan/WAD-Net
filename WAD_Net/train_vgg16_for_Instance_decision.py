import copy
import random
import os
import shutil
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.models as models
import numpy as np

from collections import OrderedDict, defaultdict
from urllib.request import urlretrieve
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True
IDRiD_path = "/home/houqs/EyeCls/datatrain_patch_withlabel_arg/"
IDRiD_file_path = sorted([os.path.join(IDRiD_path, f) for f in os.listdir(IDRiD_path)])
IDRiD_path_images_filepaths = [*IDRiD_file_path]
# correct_IDRiD_path_images_filepaths = [i for i in IDRiD_path_images_filepaths if cv2.imread(i) is not None]

random.seed(42)
# random.shuffle(correct_IDRiD_path_images_filepaths)
# train_images_filepaths = correct_IDRiD_path_images_filepaths[:2100]
# val_images_filepaths = correct_IDRiD_path_images_filepaths[2100:]
random.shuffle(IDRiD_path_images_filepaths)
train_images_filepaths = IDRiD_path_images_filepaths[:24000]
val_images_filepaths = IDRiD_path_images_filepaths[24000:]


class IDRiD2EyeQ(Dataset):
	def __init__(self, images_filepaths, transform=None):
		self.images_filepaths = images_filepaths
		self.transform = transform
	
	def __len__(self):
		return len(self.images_filepaths)
	
	def __getitem__(self, idx):
		image_filepath = self.images_filepaths[idx]
		image = cv2.imread(image_filepath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# str = os.path.normpath(image_filepath).split(os.sep)[-1][0]
		if os.path.normpath(image_filepath).split(os.sep)[-1][0] == "T":
			label = 1.0
		else:
			label = 0.0
		if self.transform is not None:
			image = self.transform(image=image)["image"]
		return image, label, image_filepath


train_transform = A.Compose(
	[
		A.SmallestMaxSize(max_size=256),
		A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
		A.RandomCrop(height=224, width=224),
		A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
		A.RandomBrightnessContrast(p=0.5),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	]
)
train_dataset = IDRiD2EyeQ(images_filepaths=train_images_filepaths, transform=train_transform)

val_transform = A.Compose(
	[
		A.SmallestMaxSize(max_size=256),
		A.CenterCrop(height=224, width=224),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	]
)
val_dataset = IDRiD2EyeQ(images_filepaths=val_images_filepaths, transform=val_transform)


def calculate_accuracy(output, target):
	output = torch.sigmoid(output) >= 0.5
	target = target == 1.0
	return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


def Cohen_kappa_score(output, target):
	output = torch.sigmoid(output).squeeze().cpu().detach()
	for i in range(len(output)):
		if output[i] > 0.5:
			output[i] = 1
		else:
			output[i] = 0
	target = target.squeeze().cpu().detach()
	kappa = cohen_kappa_score(output, target)
	return kappa


class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision
		self.reset()
	
	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
	
	def update(self, metric_name, val):
		metric = self.metrics[metric_name]
		
		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]
		return metric["avg"]
	
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
	"model": "vgg16",
	"device": "cuda",
	"lr": 0.0001,
	"batch_size": 128,
	"num_workers": 6,
	"epochs": 120,
}
##
model = getattr(models, params["model"])(pretrained=False, num_classes=1)
model_dict = model.state_dict()
model_pre = getattr(models, params["model"])(pretrained=True)
model_pre_dict = model_pre.state_dict()

new_state_dict = OrderedDict()
for k, v in model_pre_dict.items():
	if k != 'classifier.6.weight' and k != 'classifier.6.bias':
		new_state_dict[k] = v
		
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)
###########################
from torch.nn import DataParallel
model = DataParallel(model, device_ids=[0, 1])

###########################
model = model.to(params["device"])
criterion = nn.BCEWithLogitsLoss().to(params["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

train_loader = DataLoader(
	train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True
)
val_loader = DataLoader(
	val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True
)


def train(train_loader, model, criterion, optimizer, epoch, params):
	metric_monitor = MetricMonitor()
	model.train()
	stream = tqdm(train_loader)
	for i, (images, target, image_label_filepath) in enumerate(stream, start=1):
		images = images.to(params["device"], non_blocking=True)
		target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
		output = model(images)
		loss = criterion(output, target)
		kappa = Cohen_kappa_score(output, target)
		accuracy = calculate_accuracy(output, target)
		metric_monitor.update("Loss", loss.item())
		metric_monitor.update("Accuracy", accuracy)
		metric_monitor.update("kappa", kappa)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		stream.set_description(
			"Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
		)


def validate(val_loader, model, criterion, epoch, params):
	metric_monitor = MetricMonitor()
	model.eval()
	stream = tqdm(val_loader)
	with torch.no_grad():
		for i, (images, target, image_label_filepath) in enumerate(stream, start=1):
			images = images.to(params["device"], non_blocking=True)
			target = target.to(params["device"], non_blocking=True).float().view(-1, 1)
			output = model(images)
			loss = criterion(output, target)
			accuracy = calculate_accuracy(output, target)
			kappa = Cohen_kappa_score(output, target)
			loss1 = metric_monitor.update("Loss", loss.item())
			accuracy1 = metric_monitor.update("Accuracy", accuracy)
			metric_monitor.update("kappa", kappa)
			stream.set_description(
				"Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
			)
			
	return loss1, accuracy1


log_dir = "./PTH_Instance/vgg16_for_Instance_decision.pth"

if os.path.exists(log_dir):
	checkpoint = torch.load(log_dir)
	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	start_epoch = checkpoint['epoch']
	print('加载 epoch {} 成功！'.format(start_epoch))
else:
	start_epoch = 0
	print('无保存模型，将从头开始训练！')

best_metric = 0
for epoch in range(start_epoch + 1, start_epoch + 1 + params["epochs"]):
	train(train_loader, model, criterion, optimizer, epoch, params)
	loss_val, accuracy_val = validate(val_loader, model, criterion, epoch, params)
	if best_metric < accuracy_val:
		best_metric = accuracy_val
		logdir = "./PTH_Instance/vgg16_for_Instance_decision.pth"
		state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
		torch.save(state, logdir)
		print('save epoch {} success！'.format(epoch))
