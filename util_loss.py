import torch.nn as nn



#pred_x_y: b*n*2 gth_x_y: b*n*2
def loss_func(pred_x_y,gth_x_y):
	loss = nn.L1Loss()
	B,N,M= pred_x_y.shape
	z = pred_x_y[:,-1:,:]
	# print("Z",z.shape)
	xy = pred_x_y[:,:2,:]
	z = z.repeat(1,N-1,1)
	# print("pred_x_y",pred_x_y.shape,"xy",xy.shape,"z",z.shape)
	pred_x_y = xy / z
	return loss(pred_x_y,gth_x_y.cuda())