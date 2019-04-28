"""
ResNet; applied to extract feature from RGB image
Pointcloud: applied to extract feature from Point cloud
"""


import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class RGBNet(nn.Module):
	def __init__(self):
		super(RGBNet, self).__init__()
		resnet101 = models.resnet101(pretrained=True)
		modules=list(resnet101.children())[:-1]
		self.model=nn.Sequential(*modules)
		self.model.cuda()
	def forward(self, image, mesh):
		image_feature = self.model(image)
		return image_feature


if __name__ == '__main__':
	rgbNet = RGBNet()
	img = cv2.imread("test.jpg")
	img = torch.from_numpy(img).float().cuda()
	data = torch.randn(1,3,640,480).cuda()
	x = rgbNet(data,None)
	print x.shape