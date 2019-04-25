"""
0.read transformation matrix
1.read OLDmesh(point cloud of the object)
2.read R,T matrix
3.matrix multiplication
4.generate a new point cloud
5.visualization
"""

import numpy as np
import cv2
#0.read transformation matrix
Transformation = []
with open("./transform.dat") as f:
	line = f.readline()
	for _ in range(12):
		line = f.readline().split(" ")
		Transformation.append(float(line[-1]))
Transformation = np.array(Transformation,dtype=np.float64).reshape(3,4)
# Transformation = np.vstack((Transformation,np.array([[0.,0.,0.,1.]])))

#1.read a ply file
from plyfile import PlyData, PlyElement

plydata = PlyData.read("OLDmesh.ply")

#1.2save the point cloud data into a n*3 matrix
nx = np.expand_dims(plydata['vertex']['x'],0) 
ny = np.expand_dims(plydata['vertex']['y'],0) 
nz = np.expand_dims(plydata['vertex']['z'],0)
points = np.vstack((nx,ny,nz)) / 10.0
print points.shape
points = np.dot(Transformation[:,:-1],points)
points = points + Transformation[:,-1:]*100




#2.read R,T matrix
root = "./data/"
rfile = "rot0.rot"
tfile = "tra0.tra"
Rotation = np.zeros((4,3))
Translation = np.ones((4,1))
with open(root+rfile) as f:
	line = f.readline()
	for i in range(3):
		line = f.readline()
		Rotation[i] = np.array(map(float,line.split(" ")[:-1]),dtype=np.float64)
with open(root+tfile) as f:
	line = f.readline()
	for i in range(3):
		line = f.readline()
		Translation[i] = np.array(map(float,line.split(" ")[:-1]),dtype=np.float64)
print Rotation,Translation

#3.matrix multiplication
RT = np.hstack((Rotation,Translation))
points = np.vstack((points,np.ones((1,points.shape[1]))))
# points = np.dot(Transformation,points)
Plane = np.array([[572.4114, 0., 325.2611,0.],
                       [0., 573.57043, 242.04899,0.],
						[0., 0., 1.,0.]])


print Plane.shape, RT.shape, points.shape
newpoints = np.dot(Plane,np.dot(RT,points))
# newpoints = newpoints[:2,:] / newpoints[2:,:]
print newpoints.shape
# newpoints = np.vstack((newpoints[:2],np.ones((1,newpoints.shape[1]))))
#4.generate a new point cloud
#turn newpoints(3,n) into a list[n] (each point a tuple)
pointslist = []
z = np.vstack((newpoints[-1,:],newpoints[-1,:],newpoints[-1,:]))
img = cv2.imread("./color0.jpg")
for i in range(newpoints.shape[1]):
	pointslist.append(tuple(newpoints[:3,i]/z[:,i]))
	# pointslist.append(tuple(newpoints[:,i]))
	cv2.circle(img,(int(pointslist[i][0]),int(pointslist[i][1])),0,(0,0,255))
	# img[int(pointslist[i][0]),int]
cv2.imwrite("test.jpg",img)

vertex = np.array(pointslist,dtype=[('x', 'f4'), ('y', 'f4'),
                           ('z', 'f4')])
print vertex.shape
el = PlyElement.describe(vertex,'vertex') 
PlyData([el], text=True).write("newpoints.ply")               

