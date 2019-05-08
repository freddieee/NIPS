"""
Evaluation metric
1.2d projection
	compute the mean distance between the 2d gt and project, if less than < 5, consider it as a successful one
2.ADD
"""

import numpy as np
import math

#a,b (2,)
def ec_dis(a,b):
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

#prediction: B*N*2 gth:B*N*2
def evaluate_projection(predicton,gth):

	B,N,M= predicton.shape
	z = predicton[:,-1:,:]
	# print("Z",z.shape)
	xy = predicton[:,:2,:]
	z = z.repeat(1,N-1,1)
	# print("pred_x_y",pred_x_y.shape,"xy",xy.shape,"z",z.shape)
	predicton = xy / z
	batchsize = predicton.shape[0]
	num_of_points = predicton.shape[1]
	positive = 0
	print("predicton:",predicton.shape)
	print("gth",gth.shape)
	print("predicton",predicton[0])
	for b in range(batchsize):
		totaldistance = sum([ec_dis(predicton[b][i],gth[b][i]) for i in range(num_of_points)])
		meandistance = totaldistance / num_of_points
		if meandistance<=5.0:
			positive += 1

	return positive, 1.0*positive/batchsize



