"""
1.for init:
	set a list for train/test
"""

import torch
from torch.utils import data
import os
import cv2
from plyfile import PlyData, PlyElement
import numpy as np
class Linemod_Dataset(data.Dataset):
  def __init__(self, train = True,root="../linemod/"):
		self.root = root
		self.objs = ["ape","benchsive","cam","can","cat","driller","duck","eggbox","glue","holepuncher","iron","lamp","phone"]
		self.list_IDs = []
		for obj in self.objs[:1]:
			dir_obj = self.root+obj+"/data"
			nums = len([name for name in os.listdir(dir_obj) if os.path.isfile(os.path.join(dir_obj, name))]) / 4
			pointcloud = readPointcloud(self.root+obj)
			if train:
				for i in range(nums*4/5):
					dir_image = dir_obj+"/color{0}.jpg".format(i)
					self.list_IDs.append((i,obj,pointcloud,dir_image))
			else:
				for i in range(nums*4/5,nums):
					dir_image = dir_obj+"/color{0}.jpg".format(i)
					self.list_IDs.append((i,obj,pointcloud,dir_image))	        		
		

  def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)


  #return id,model_cls,pointcloud,rgb
  def __getitem__(self, index):
		'Generates one sample of data'
		id, model_cls, pointcloud, rgb_path = self.list_IDs[index]
		#read rgb file and turn it into 3*640*480 tensor
		rgb = readRGB(rgb_path)
		return id,model_cls,pointcloud,rgb


#read the pointcloud and return a n*3 tensor
def readPointcloud(root):	
	#0.read transformation matrix
	Transformation = []
	with open(root+"/transform.dat") as f:
		line = f.readline()
		for _ in range(12):
			line = f.readline().split(" ")
			Transformation.append(float(line[-1]))
	Transformation = np.array(Transformation,dtype=np.float64).reshape(3,4)
	plydata = PlyData.read(root+"/OLDmesh.ply")
	#1.2save the point cloud data into a n*3 matrix
	nx = np.expand_dims(plydata['vertex']['x'],0) 
	ny = np.expand_dims(plydata['vertex']['y'],0) 
	nz = np.expand_dims(plydata['vertex']['z'],0)
	points = np.vstack((nx,ny,nz)) / 10.0
	points = np.dot(Transformation[:,:-1],points)
	points = points + Transformation[:,-1:]*100
	return points
#read the RGB and retuen a 3*640*480 tensor
def readRGB(rgb_path):
	img = cv2.imread(rgb_path)
	img = cv2.transpose(img,(0,1,2,3))
	return img.transpose((2,0,1))


if __name__ == '__main__':
	dataset = Linemod_Dataset()
	print len(dataset),dataset[1]