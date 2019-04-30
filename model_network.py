"""
ResNet; applied to extract feature from RGB image
Pointcloud: applied to extract feature from Point cloud
"""


import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
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

class STN3d(nn.Module):
	def __init__(self, num_points = 2500):
		super(STN3d, self).__init__()
		self.num_points = num_points
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 1024, 1)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 9)
		self.relu = nn.ReLU()

	def forward(self, x):
		batchsize = x.size()[0]
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x,_ = torch.max(x, 2)
		x = x.view(-1, 1024)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, 3, 3)
		return x

#input:B*3*2048 output:B*1024
class PointNetfeat(nn.Module):
	def __init__(self, num_points = 2048, global_feat = True, trans = False):
		super(PointNetfeat, self).__init__()
		self.stn = STN3d(num_points = num_points)
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 1024, 1)

		self.bn1 = torch.nn.BatchNorm1d(64)
		self.bn2 = torch.nn.BatchNorm1d(128)
		self.bn3 = torch.nn.BatchNorm1d(1024)
		self.trans = trans


		self.num_points = num_points
		self.global_feat = global_feat
	def forward(self, x):
		batchsize = x.size()[0]
		if self.trans:
			trans = self.stn(x)
			x = x.transpose(2,1)
			x = torch.bmm(x, trans)
			x = x.transpose(2,1)
		x = F.relu(self.bn1(self.conv1(x)))
		pointfeat = x
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x,_ = torch.max(x, 2)
		x = x.view(-1, 1024)
		if self.trans:
			if self.global_feat:
				return x, trans
			else:
				x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
				return torch.cat([x, pointfeat], 1), trans
		else:
			return x

class FeatureRegression(nn.Module):
    def __init__(self, inputdim=2048,bottleneck=1024):
        super(FeatureRegression, self).__init__()
        self.inputdim=inputdim
        self.conv = nn.Sequential(
            nn.Conv1d(self.inputdim, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256,1,1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ImageNet(torch.nn.Module):
    def __init__(self, use_cuda=True, feature_extraction_cnn='resnet101', last_layer=''):
        super(ImageNet, self).__init__()
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model.cuda()

    #input size: B*3*640*480 outputsize: B*1024*40*30
    def forward(self, image_batch):
        return self.model(image_batch)


class ContinuousField(nn.Module):
    def __init__(self):
        super(ContinuousField, self).__init__()
        self.fg = FeatureRegression()
        self.net = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1),
        )

        self.th = nn.Tanh()

     #x: B*2048*1024 ,pc:B*3*2048 output:B*3*N
    def forward(self,pc,x):
    	x = self.fg(x)  #B*1*1024
    	x = x.transpose(1,2) #B*1024*1
    	x = x.repeat(1,1,2048)
    	print x.shape,pc.shape
    	x = torch.cat((pc,x),1)
    	x = self.net(x)
    	x = self.th(x)
    	return x

#pc_feature:B*1024 rgb_feature:B*1024*32*32 output = B*2048*1024	
def intergrateFeature(pc_feature,rgb_feature):
	return pc_feature+rgb_feature

if __name__ == '__main__':
	imgNet = ImageNet()
	img = cv2.imread("test.jpg")
	regression = FeatureRegression().cuda()
	img = torch.from_numpy(img).float().cuda()
	data = torch.randn(1,3,2048).cuda()
	# data = torch.randn(1,2048,1024).cuda()
	# tp = torch.randn(1,3,2048).cuda()
	# cf = ContinuousField().cuda()
	# x = cf(tp,data)
	# fr = FeatureRegression().cuda()
	# x = fr(data)
	# z = imgNet(data)
	pointNet = PointNetfeat().cuda()
	# data = torch.randn(1,3,2048).float().cuda()
	y = pointNet(data)
	print y.shape