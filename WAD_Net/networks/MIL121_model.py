import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import re

from torch.jit.annotations import List
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class _DenseLayer(nn.Module):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
		self.add_module('relu1', nn.ReLU(inplace=True)),
		self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
		                                   growth_rate, kernel_size=1, stride=1,
		                                   bias=False)),
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
		self.add_module('relu2', nn.ReLU(inplace=True)),
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
		                                   kernel_size=3, stride=1, padding=1,
		                                   bias=False)),
		self.drop_rate = float(drop_rate)
		self.memory_efficient = memory_efficient
	
	def bn_function(self, inputs):
		# type: (List[Tensor]) -> Tensor
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
		return bottleneck_output
	
	# todo: rewrite when torchscript supports any
	def any_requires_grad(self, input):
		# type: (List[Tensor]) -> bool
		for tensor in input:
			if tensor.requires_grad:
				return True
		return False
	
	@torch.jit.unused  # noqa: T484
	def call_checkpoint_bottleneck(self, input):
		# type: (List[Tensor]) -> Tensor
		def closure(*inputs):
			return self.bn_function(inputs)
		
		return cp.checkpoint(closure, *input)
	
	@torch.jit._overload_method  # noqa: F811
	def forward(self, input):
		# type: (List[Tensor]) -> (Tensor)
		pass
	
	@torch.jit._overload_method  # noqa: F811
	def forward(self, input):
		# type: (Tensor) -> (Tensor)
		pass
	
	# torchscript does not yet support *args, so we overload method
	# allowing it to take either a List[Tensor] or single Tensor
	def forward(self, input):  # noqa: F811
		if isinstance(input, Tensor):
			prev_features = [input]
		else:
			prev_features = input
		
		if self.memory_efficient and self.any_requires_grad(prev_features):
			if torch.jit.is_scripting():
				raise Exception("Memory Efficient not supported in JIT")
			
			bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
		else:
			bottleneck_output = self.bn_function(prev_features)
		
		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate,
			                         training=self.training)
		return new_features


class _DenseBlock(nn.ModuleDict):
	_version = 2
	
	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(
				num_input_features + i * growth_rate,
				growth_rate=growth_rate,
				bn_size=bn_size,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient,
			)
			self.add_module('denselayer%d' % (i + 1), layer)
	
	def forward(self, init_features):
		features = [init_features]
		for name, layer in self.items():
			new_features = layer(features)
			features.append(new_features)
		return torch.cat(features, 1)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(num_input_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
		                                  kernel_size=1, stride=1, bias=False))
		self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MIL_model(nn.Module):
	def __init__(self,
		layers,
	    num_classes = 5,
		num_feat = 1024,
	    dropout = 0
	):
		
		super(MIL_model, self).__init__()
		self.L = 512
		self.D = 256
		self.K = 5
		
		# self.feature_extractor_part1 = nn.Sequential(
		# 	nn.Conv2d(256, 128, kernel_size=3),
		# 	nn.ReLU(),
		# 	nn.MaxPool2d(2, stride=2),
		# 	nn.Conv2d(128, 50, kernel_size=3),
		# 	nn.ReLU(),
		# 	nn.MaxPool2d(2, stride=2)
		# )
		#
		# self.feature_extractor_part2 = nn.Sequential(
		# 	nn.Linear(50 * 6 * 6, self.L),
		# 	nn.ReLU(),
		# )
		
		self.attention = nn.Sequential(
			nn.Linear(self.L, self.D),
			nn.Tanh(),
			nn.Linear(self.D, self.K)
		)
		
		# self.classifier = nn.Sequential(
		# 	nn.Linear(self.L * self.K, 1),
		# 	nn.Sigmoid()
		# )
		#####################################
		self.num_classes = num_classes
		self.num_feat = num_feat
		self.dropout = dropout
		
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2,
			                    padding=3, bias=False)),
			('norm0', nn.BatchNorm2d(64)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))
		
		Denseblock1 = _DenseBlock(
			num_layers=layers[0],
			num_input_features=64,
			bn_size=4,
			growth_rate=32,
			drop_rate=0,
			memory_efficient=False
		)
		self.features.denseblock1 = Denseblock1
		transition1 = _Transition(num_input_features=256, num_output_features=128)
		self.features.transition1 = transition1
		Denseblock2 = _DenseBlock(
			num_layers=layers[1],
			num_input_features=128,
			bn_size=4,
			growth_rate=32,
			drop_rate=0,
			memory_efficient=False
		)
		self.features.denseblock2 = Denseblock2
		transition2 = _Transition(num_input_features=512, num_output_features=256)
		self.features.transition2 = transition2

		Denseblock3 = _DenseBlock(
			num_layers=layers[2],
			num_input_features=256,
			bn_size=4,
			growth_rate=32,
			drop_rate=0,
			memory_efficient=False
		)
		self.features.denseblock3 = Denseblock3
		transition3 = _Transition(num_input_features=1024, num_output_features=512)
		self.features.transition3 = transition3

		Denseblock4 = _DenseBlock(
			num_layers=layers[3],
			num_input_features=512,
			bn_size=4,
			growth_rate=32,
			drop_rate=0,
			memory_efficient=False
		)
		self.features.denseblock4 = Denseblock4
		self.features.norm5 = nn.BatchNorm2d(1024)
		if self.dropout > 0:
			self.drop = nn.Dropout(self.dropout)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(36*5*512, 2048)
		self.cls = nn.Linear(2048, self.num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)
				
	def forward(self, x):
		x1 = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=128, p2=128)
		x1 = self.features(x1)
		x1 = F.relu(x1, inplace=True)
		feat_ = F.adaptive_avg_pool2d(x1, (1, 1))
		feat = torch.flatten(feat_, 1)
		if self.dropout > 0:
			feat = self.drop(feat)
		feat = self.fc1(feat)
		feat = self.fc2(feat)
		feat1 = feat.unsqueeze(1)
		
		A = self.attention(feat)
		A = F.softmax(A, dim=0)
		A1 = A.unsqueeze(-1)
		
		featA = torch.matmul(A1, feat1) # 72 5 512
		featA = featA.view(-1, 5*512)
		featA1 = rearrange(featA, '(b p) L -> b (p L)', b=x.shape[0])
		featA1 = self.fc3(featA1)
		featA1 = self.cls(featA1)
		
		return featA1


def Create_MIL_model(layers=[6, 12, 24, 16], num_classes=5):
	model = MIL_model(layers=layers, num_classes=num_classes)
	
	return model