"""
for visualization, we draw 8 corners points for both groundtruth and our trained mapping function

1. get 8 corner points in 3D space for given model
2. for ground truth, we use tranform matri, RT matrix and intrisic matrix to map the points on RGB plane
3. for trained mapping function, we predict (x,y) directly for each point 
"""
from util_Projection import project_3d_on_2d
from plyfile import PlyData, PlyElement
import numpy as np
import cv2
processed_data = "../PoseEstimation/data/LINEMOD/"
linemod_data = "../linemod/"
def get_corners(obj):
	plydata = PlyData.read(linemod_data+obj+"/OLDmesh.ply")

	nx = plydata['vertex']['x'] / 10.0
	ny = plydata['vertex']['y'] / 10.0
	nz = plydata['vertex']['z'] / 10.0

	X_max,X_min = max(nx),min(nx)
	Y_max,Y_min = max(ny),min(ny)
	Z_max,Z_min = max(nz),min(nz)

	corners = [[X_min,Y_max,Z_max],[X_max,Y_max,Z_max],[X_max,Y_max,Z_min],[X_min,Y_max,Z_min],
	[X_min,Y_min,Z_max],[X_max,Y_min,Z_max],[X_max,Y_min,Z_min],[X_min,Y_min,Z_min]]

	return np.array(corners).T



def draw_gth_RGB(obj="ape",id=0):
	image_path = linemod_data+obj+"/data/color{0}.jpg".format(id)
	corners = get_corners(obj)

	newpoints = project_3d_on_2d(obj,id,corners)
	img = cv2.imread(image_path)
	pointslist = []
	z = np.vstack((newpoints[-1,:],newpoints[-1,:],newpoints[-1,:]))
	for i in range(newpoints.shape[1]):
		pointslist.append(tuple(newpoints[:3,i]/z[:,i]))

		# pointslist.append(tuple(newpoints[:,i]))
		# cv2.circle(img,(int(pointslist[i][0]),int(pointslist[i][1])),5,(0,0,255))
		# img[int(pointslist[i][0]),int]
	red = (0,0,255)
	lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(0,4),(1,5),(2,6),(3,7),(7,4)]
	for line in lines:
		start,end = line
		cv2.line(img,(int(pointslist[start][0]),int(pointslist[start][1])),(int(pointslist[end][0]),int(pointslist[end][1])),red)

	# #1.read a ply file
	
	# plydata = PlyData.read(linemod_data+obj+"/OLDmesh.ply")

	# #1.2save the point cloud data into a n*3 matrix
	# nx = np.expand_dims(plydata['vertex']['x'],0) 
	# ny = np.expand_dims(plydata['vertex']['y'],0) 
	# nz = np.expand_dims(plydata['vertex']['z'],0)
	# points = np.vstack((nx,ny,nz)) / 10.0
	# newpoints = project_3d_on_2d(obj,id,points)
	# pointslist = []
	# z = np.vstack((newpoints[-1,:],newpoints[-1,:],newpoints[-1,:]))
	# for i in range(newpoints.shape[1]):
	# 	pointslist.append(tuple(newpoints[:3,i]/z[:,i]))
	# 	# pointslist.append(tuple(newpoints[:,i]))
	# 	cv2.circle(img,(int(pointslist[i][0]),int(pointslist[i][1])),5,(0,0,255))
	cv2.imwrite("color{0}.jpg".format(id),img)


if __name__ == '__main__':
	draw_gth_RGB("ape", 1008)