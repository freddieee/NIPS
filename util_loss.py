import torch.nn as nn



#pred_x_y: b*n*2 gth_x_y: b*n*2
def loss_func(pred_x_y,gth_x_y):
	return nn.L1Loss(pred_x_y,gth)