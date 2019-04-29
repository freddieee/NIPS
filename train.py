"""
1.set the parse
2.set the dataset
3.set the network
4.run epoch
	5.1 for each pair(RGB, model),compute the the total loss
	5.2 backward
5.save the modeldict
"""

from __future__ import print_function, division
import os,sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--worker', type=int, default=1, help='data loader worker')
parser.add_argument('--nepoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--ngpu', type=int, default=4, help='number of gpus')
parser.add_argument('--main_gpu', type=int, default=0, help='main gpu')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.0005, help='momentum constant')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='momentum constant')

opt = parser.parse_args()
print(opt)

linemod_train = Linemod_Dataset(train=True,root="../linemod/")
linemod_test = Linemod_Dataset(train=False,root="../linemod/")

train_loader = torch.utils.data.DataLoader(linemod_train, batch_size=opt.batchSize,
										  shuffle=True, num_workers=int(opt.worker), pin_memory=False)

test_loader = torch.utils.data.DataLoader(linemod_test, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.worker), pin_memory=False)

#set the network
imgNet = ImageNet()
pointNet = PointNetfeat().cuda()
continuousField = ContinuousField().cuda()

optimizer = optim.Adam(continuousField.parameters(), lr=1e-3, betas=(0.5, 0.999), eps=1e-06)

continuousField.train()
pointNet.train()
#run the epoch
for epoch in range(nepoch):
	total_loss = 0
	for id,model_cls,pointcloud,rgb in enumerate(train_loader):
		pc_feature = pointNet(pointcloud)
		rgb_feature = imgNet()
		intergratingFeature = intergratingFeature(pc_feature,rgb_feature)
		pred_x_y = continuousField(pointcloud,intergratingFeature)
		gth_x_y = project_3d_on_2d(obj=model_cls,id,pointcloud)
		loss = loss_func(pred_x_y,gth_x_y)
		total += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	if epoch % 5 == 0:
		torch.save(pointNet.state_dict(),"weigths/pointNet/{epoch}.pth")
		torch.save(continuousField.state_dict(),"weigths/continuousField/{epoch}.pth")
	print("Epoch {0} finished, total loss:{1}".format(epoch,total_loss))
	print("Evaluating")
	positives = 0
	for id,model_cls,pointcloud,rgb in enumerate(test_loader):
		pc_feature = pointNet(pointcloud)
		rgb_feature = imgNet()
		intergratingFeature = intergratingFeature(pc_feature,rgb_feature)
		pred_x_y = continuousField(pointcloud,intergratingFeature)
		gth_x_y = project_3d_on_2d(obj=model_cls,id,pointcloud)
		positive,accuracy = evaluate_projection(pred_x_y,gth_x_y)
		positives += positive 
	print("Epoch{0} :accuracy is {:.2%}%".format(epoch, positive/len(test_loader)))

