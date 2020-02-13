import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tensorboardX import SummaryWriter

class modelResnet18(nn.Module):
	"""Atrous Spatial Pyramid Pooling"""

	def __init__(self,num_classes = 1):

		super(modelResnet18, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		NumofFeaturesLastLayer=self.resnet18.fc.in_features
		newLastLayer=nn.Linear(NumofFeaturesLastLayer, 4)
		self.resnet18.fc=newLastLayer


	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

	def forward(self, input):
		
		return self.resnet18.forward(input)

	def loss(self,output, target):
		criterion = nn.SmoothL1Loss()
		loss = criterion(output, target)  # self.loss ro baayad bargardoonam 
		return loss

