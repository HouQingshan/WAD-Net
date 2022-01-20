import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
	                 padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion: int = 1
	
	def __init__(
			self,
			inplanes: int,
			planes: int,
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
			norm_layer: Optional[Callable[..., nn.Module]] = None
	) -> None:
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x: Tensor) -> Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.relu(out)
		
		return out


class Bottleneck(nn.Module):
	expansion: int = 4
	
	def __init__(
			self,
			inplanes: int,
			planes: int,
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
			norm_layer: Optional[Callable[..., nn.Module]] = None
	) -> None:
		
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = planes
		
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x: Tensor) -> Tensor:
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.relu(out)
		
		return out


class MIL_model(nn.Module):
	def __init__(self,
		block,
		layers,
	    num_classes,
		norm_layer = None
	):
		
		super(MIL_model, self).__init__()
		self.L = 500
		self.D = 128
		self.K = 1
		
		self.feature_extractor_part1 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(128, 50, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2)
		)
		
		self.feature_extractor_part2 = nn.Sequential(
			nn.Linear(50 * 6 * 6, self.L),
			nn.ReLU(),
		)
		
		self.attention = nn.Sequential(
			nn.Linear(self.L, self.D),
			nn.Tanh(),
			nn.Linear(self.D, self.K)
		)
		
		self.classifier = nn.Sequential(
			nn.Linear(self.L * self.K, 1),
			nn.Sigmoid()
		)
		
		
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		
		self._norm_layer = norm_layer
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
	                stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
		
		self.inplanes = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x) # 2 64 384 384
		x = self.maxpool(x) # 2 64 192 192
		
		x = self.layer1(x) # 2 256 192 192
		x1 = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=32, p2=32) # 72 256 32 32
		H = self.feature_extractor_part1(x1) # 72 50 6 6
		H = H.view(-1, 50 * 6 * 6) # 72 1800
		H = self.feature_extractor_part2(H) # 72 500
		A = self.attention(H) # 72 1
		A = F.softmax(A, dim=0)
		x_patch = x1.clone()
		for i in range(len(A)):
			if A[i] == 0:
				pass
			else:
				x_patch[i, :, :, :] = x1[i, :, :, :] * A[i]
				# print(x1[i, :, :, :])
				# print(x_patch[i, :, :, :])
				# print(A[i])
			
		images_weight = rearrange(x_patch, '(b h w) c p1 p2 -> b c (h p1) (w p2)', b=x.shape[0],
		                         h=6)
		
		
		images_weight = self.layer2(images_weight) # 2 512 96 96
		images_weight = self.layer3(images_weight) # 2 1024 48 48
		images_weight = self.layer4(images_weight)# 2 2048 24 24
		images_weight = self.avgpool(images_weight)# 2 2048 1 1
		images_weight = torch.flatten(images_weight, 1)# 2 2048
		images_weight = self.fc(images_weight) # 2 5
		
		return images_weight


def Create_MIL_model(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=5):
	model = MIL_model(block=block, layers=layers, num_classes=num_classes)
	
	return model