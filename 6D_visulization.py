"""
for visualization, we draw 8 corners points for both groundtruth and our trained mapping function

1. get 8 corner points in 3D space for given model
2. for ground truth, we use tranform matri, RT matrix and intrisic matrix to map the points on RGB plane
3. for trained mapping function, we predict (x,y) directly for each point 
"""
from 6D_1 import project_3d_on_2d

processed_data = "../PoseEstimation/data/LINEMOD/"
linemod_data = "../linemod/"
def get_corners(obj):
	file = processed_data+obj+"/corners.txt"
	corners = []
	with open(file) as f:
		for i in range(8):
			line = f.readline().split(" ")
			corner = map(float,line)
			corner = [i*100 for i in corner]
			corners.append(corner)

	return np.array(corners).T

print get_corners("ape")

import numpy as np
def draw_gth_RGB(obj="ape",id=0):
	image_path = linemod+obj+"/data/color{0}.jpg".format(id)
	corners = get_corners(obj)
	newpoints = project_3d_on_2d(obj,id,corners)
	img = cv2.imread(image_path)
	for i in range(newpoints.shape[1]):
		pointslist.append(tuple(newpoints[:3,i]/z[:,i]))
		# pointslist.append(tuple(newpoints[:,i]))
		cv2.circle(img,(int(pointslist[i][0]),int(pointslist[i][1])),0,(0,0,255))
		# img[int(pointslist[i][0]),int]
	cv2.imwrite("test.jpg",img)


if __name__ == '__main__':
	draw_gth_RGB("ape", 100)