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
from util_loss import loss_func
from data_linemod import Linemod_Dataset
from model_network import ImageNet, PointNetfeat, ContinuousField, Network_Util
import torch.optim as optim
from util_Projection import project_3d_on_2d
from model_evaluation import evaluate_projection
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--worker', type=int, default=0, help='data loader worker')
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

optimizer = optim.Adam(pointNet.parameters(), lr=1e-3, betas=(0.5, 0.999), eps=1e-06)
optimizer_continuousField = optim.Adam(continuousField.parameters(), lr=1e-3, betas=(0.5, 0.999), eps=1e-06)
continuousField.train()
pointNet.train()
#run the epoch\
iteration = 0
for epoch in range(opt.nepoch):
	total_loss = 0



	for index,x in enumerate(train_loader):
		id,gth_x_y,pointcloud,rgb = x
		rgb = Variable(rgb).float().cuda()
		pointcloud = Variable(pointcloud).float().cuda()

		pc_feature = pointNet(pointcloud)
		rgb_feature = imgNet(rgb)
		intergratingFeature = Network_Util.intergrateFeature(pc_feature,rgb_feature)
		pred_x_y = continuousField(pointcloud,intergratingFeature)

		loss = loss_func(pred_x_y,gth_x_y)
		iteration = iteration+opt.batchSize
		print("loss:",loss)
		print("epoch:{0} iter:{1} Loss:{2}".format(epoch,iteration,loss))
		total_loss += loss
		optimizer.zero_grad()
		optimizer_continuousField.zero_grad()
		loss.backward()
		optimizer.step()
		optimizer.step()

	if epoch % 10 == 0:
		torch.save(pointNet.state_dict(),"weights/pointNet/{epoch}.pth")
		torch.save(continuousField.state_dict(),"weights/continuousField/{epoch}.pth")

		print("Evaluating")
	positives = 0
	for index,x in enumerate(test_loader):
		id,gth_x_y,pointcloud,rgb = x
		rgb = Variable(rgb).float().cuda()
		pointcloud = Variable(pointcloud).float().cuda()
		pc_feature = pointNet(pointcloud)
		rgb_feature = imgNet(rgb)
		intergratingFeature = Network_Util.intergrateFeature(pc_feature,rgb_feature)
		pred_x_y = continuousField(pointcloud,intergratingFeature)
		positive,accuracy = evaluate_projection(pred_x_y,gth_x_y)
		positives += positive 
	print("Epoch{} :positives:{} accuracy is {:.2%}%".format(epoch, positives,1.0*positive/len(test_loader)))
	print("Epoch {0} finished, total loss:{1}".format(epoch,total_loss))


