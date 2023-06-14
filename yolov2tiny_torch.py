import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate
from loss import build_target, yolo_loss
from cnn_torch import *
from pathlib import Path

def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

class WeightLoader(object):
	def __init__(self):
		super(WeightLoader, self).__init__()
		self.start = 0
		self.buf = None
		self.b = 'b'
		self.g = 'g'
		self.rm = 'rm'
		self.rv = 'rv'
		
	def load_conv_bn(self, conv_model, bn_model):

		# Make directories
		Path('./weight_parameter/bn_param/bias').mkdir(parents=True, exist_ok=True)
		Path('./weight_parameter/bn_param/gamma').mkdir(parents=True, exist_ok=True)
		Path('./weight_parameter/bn_param/running_mean').mkdir(parents=True, exist_ok=True)
		Path('./weight_parameter/bn_param/running_var').mkdir(parents=True, exist_ok=True)
		Path('./weight_parameter/conv_param/w').mkdir(parents=True, exist_ok=True)
		Path('./weight_parameter/bias/').mkdir(parents=True, exist_ok=True)
		
		
		num_w = conv_model.weight.numel()
		num_b = bn_model.bias.numel()

		bn_model.bias.data.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

		if bn_model.bias.data.shape == self.scratch.params['beta0'].shape:
			self.scratch.params['beta0'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(0), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif bn_model.bias.data.shape == self.scratch.params['beta1'].shape:
			self.scratch.params['beta1'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(1), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif bn_model.bias.data.shape == self.scratch.params['beta2'].shape:
			self.scratch.params['beta2'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(2), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif bn_model.bias.data.shape == self.scratch.params['beta3'].shape:
			self.scratch.params['beta3'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(3), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif bn_model.bias.data.shape == self.scratch.params['beta4'].shape:
			self.scratch.params['beta4'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(4), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif bn_model.bias.data.shape == self.scratch.params['beta5'].shape:
			self.scratch.params['beta5'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(5), mode='w') as f:
				f.write(str(bn_model.bias.data))
		elif (bn_model.bias.data.shape == self.scratch.params['beta6'].shape) and self.b == "b":
			self.scratch.params['beta6'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(6), mode='w') as f:
				f.write(str(bn_model.bias.data))
			self.b = 'bb'
		elif (self.scratch.params['beta7'].shape == bn_model.bias.data.shape) and self.b == "bb":
			self.scratch.params['beta7'] = bn_model.bias.data
			with open('./weight_parameter/bn_param/bias/{}'.format(7), mode='w') as f:
				f.write(str(bn_model.bias.data))

		self.start = self.start + num_b

		

		bn_model.weight.data.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))


		if bn_model.weight.data.shape == self.scratch.params['gamma0'].shape:
			self.scratch.params['gamma0'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(0), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif bn_model.weight.data.shape == self.scratch.params['gamma1'].shape:
			self.scratch.params['gamma1'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(1), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif bn_model.weight.data.shape == self.scratch.params['gamma2'].shape:
			self.scratch.params['gamma2'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(2), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif bn_model.weight.data.shape == self.scratch.params['gamma3'].shape:
			self.scratch.params['gamma3'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(3), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif bn_model.weight.data.shape == self.scratch.params['gamma4'].shape:
			self.scratch.params['gamma4'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(4), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif bn_model.weight.data.shape == self.scratch.params['gamma5'].shape:
			self.scratch.params['gamma5'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(5), mode='w') as f:
				f.write(str(bn_model.weight.data))
		elif (bn_model.weight.shape == self.scratch.params['gamma6'].shape) and self.g == "g":
			self.scratch.params['gamma6'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(6), mode='w') as f:
				f.write(str(bn_model.weight.data))
			self.g = 'gg'
		elif (self.scratch.params['gamma7'].shape == bn_model.weight.data.shape) and self.g == "gg":
			self.scratch.params['gamma7'] = bn_model.weight.data
			with open('./weight_parameter/bn_param/gamma/{}'.format(7), mode='w') as f:
				f.write(str(bn_model.weight.data))

		self.start = self.start + num_b

		bn_model.running_mean.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

		if bn_model.running_mean.data.shape == self.scratch.bn_params[0]['running_mean'].shape:
			self.scratch.bn_params[0]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(0), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[1]['running_mean'].shape:
			self.scratch.bn_params[1]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(1), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[2]['running_mean'].shape:
			self.scratch.bn_params[2]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(2), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[3]['running_mean'].shape:
			self.scratch.bn_params[3]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(3), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[4]['running_mean'].shape:
			self.scratch.bn_params[4]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(4), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[5]['running_mean'].shape:
			self.scratch.bn_params[5]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(5), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[6]['running_mean'].shape and self.rm == "rm":
			self.scratch.bn_params[6]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(6), mode='w') as f:
				f.write(str(bn_model.running_mean.data))
			self.rm = "rmrm"
		elif bn_model.running_mean.data.shape == self.scratch.bn_params[7]['running_mean'].shape and self.rm == "rmrm":
			self.scratch.bn_params[7]['running_mean'] = bn_model.running_mean.data
			with open('./weight_parameter/bn_param/running_mean/{}'.format(7), mode='w') as f:
				f.write(str(bn_model.running_mean.data))

		self.start = self.start + num_b

		bn_model.running_var.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

		if bn_model.running_var.data.shape == self.scratch.bn_params[0]['running_var'].shape:
			self.scratch.bn_params[0]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(0), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[1]['running_var'].shape:
			self.scratch.bn_params[1]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(1), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[2]['running_var'].shape:
			self.scratch.bn_params[2]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(2), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[3]['running_var'].shape:
			self.scratch.bn_params[3]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(3), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[4]['running_var'].shape:
			self.scratch.bn_params[4]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(4), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[5]['running_var'].shape:
			self.scratch.bn_params[5]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(5), mode='w') as f:
				f.write(str(bn_model.running_var.data))
		elif bn_model.running_var.data.shape == self.scratch.bn_params[6]['running_var'].shape and self.rv == "rv":
			self.scratch.bn_params[6]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(6), mode='w') as f:
				f.write(str(bn_model.running_var.data))
			self.rv = "rvrv"
		elif bn_model.running_var.data.shape == self.scratch.bn_params[7]['running_var'].shape and self.rv == "rvrv":
			self.scratch.bn_params[7]['running_var'] = bn_model.running_var.data
			with open('./weight_parameter/bn_param/running_var/{}'.format(7), mode='w') as f:
				f.write(str(bn_model.running_var.data))
			
		self.start = self.start + num_b

		conv_model.weight.data.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
		
		
		if conv_model.weight.data.shape == (16, 3, 3, 3):
			self.scratch.params['W0'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(0), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (32, 16, 3, 3):
			self.scratch.params['W1'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(1), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (64, 32, 3, 3):
			self.scratch.params['W2'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(2), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (128, 64, 3, 3):
			self.scratch.params['W3'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(3), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (256, 128, 3, 3):
			self.scratch.params['W4'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(4), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (512, 256, 3, 3):
			self.scratch.params['W5'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(5), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (1024, 512, 3, 3):            
			self.scratch.params['W6'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(6), mode='w') as f:
				f.write(str(conv_model.weight.data))
		elif conv_model.weight.data.shape == (1024, 1024, 3, 3):
			self.scratch.params['W7'] = conv_model.weight.data
			with open('./weight_parameter/conv_param/w/{}'.format(7), mode='w') as f:
				f.write(str(conv_model.weight.data))
		self.start = self.start + num_w

	def load_conv(self, conv_model):
		num_w = conv_model.weight.numel()
		num_b = conv_model.bias.numel()
		conv_model.bias.data.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
		self.scratch.params['b8'] = conv_model.bias.data
		with open('./weight_parameter/bias/{}'.format(7), mode='w') as f:
			f.write(str(conv_model.bias.data))
		self.start = self.start + num_b
		conv_model.weight.data.copy_(
			torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
		self.scratch.params['W8'] = conv_model.weight.data
		with open('./weight_parameter/conv_param/w/{}'.format(8), mode='w') as f:
			f.write(str(conv_model.weight.data))
		self.start = self.start + num_w

	def dfs(self, m):
		children = list(m.children())
		for i, c in enumerate(children):
			if isinstance(c, torch.nn.Sequential):
				self.dfs(c)
			elif isinstance(c, torch.nn.Conv2d):
				if c.bias is not None:
					self.load_conv(c)
				else:
					self.load_conv_bn(c, children[i + 1])

	def load(self, model_to_load_weights_to, model, weights_file):
		self.scratch = model_to_load_weights_to 
		self.start = 0
		fp = open(weights_file, 'rb')
		header = np.fromfile(fp, count=4, dtype=np.int32)
		self.buf = np.fromfile(fp, dtype=np.float32)
		fp.close()
		size = self.buf.size
		self.dfs(model)
		# make sure the loaded weight is right
		assert size == self.start
		return self.scratch

pytorch_model = DeepConvNetTorch    (input_dims=(3, 416, 416),
                                    num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                    max_pools=[0, 1, 2, 3, 4],
                                    weight_scale='kaiming',
                                    batchnorm=True,
                                    dtype=torch.float32, device='cpu')
  
_Load_Weights = True
if _Load_Weights:
	class Yolov2(nn.Module):

		num_classes = 20
		num_anchors = 5

		def __init__(self, classes=None, weights_file=False):
			super(Yolov2, self).__init__()
			if classes:
				self.num_classes = len(classes)


			self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
			self.lrelu = nn.LeakyReLU(0.1, inplace=True)
			self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

			
			self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn1 = nn.BatchNorm2d(16)

			self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn2 = nn.BatchNorm2d(32)

			self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn3 = nn.BatchNorm2d(64)

			self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn4 = nn.BatchNorm2d(128)

			self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn5 = nn.BatchNorm2d(256)

			self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn6 = nn.BatchNorm2d(512)

			self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn7 = nn.BatchNorm2d(1024)

			self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn8 = nn.BatchNorm2d(1024)

			self.conv9 = nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)


		def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
			"""
			x: Variable
			gt_boxes, gt_classes, num_boxes: Tensor
			"""

			x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
			x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
			x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
			x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
			x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
			x = self.lrelu(self.bn6(self.conv6(x)))
			x = F.pad(x, (0, 1, 0, 1))
			x = self.slowpool(x)
			x = self.lrelu(self.bn7(self.conv7(x)))
			x = self.lrelu(self.bn8(self.conv8(x)))
			out = self.conv9(x)



			# out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
			bsize, _, h, w = out.size()

			# 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
			# reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
			out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

			# activate the output tensor
			# `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
			# `softmax` for (class1_score, class2_score, ...)

			xy_pred = torch.sigmoid(out[:, :, 0:2])
			conf_pred = torch.sigmoid(out[:, :, 4:5])
			hw_pred = torch.exp(out[:, :, 2:4])
			class_score = out[:, :, 5:]
			class_pred = F.softmax(class_score, dim=-1)
			delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

			if training:
				output_variable = (delta_pred, conf_pred, class_score)
				output_data = [v.data for v in output_variable]
				gt_data = (gt_boxes, gt_classes, num_boxes)
				target_data = build_target(output_data, gt_data, h, w)

				target_variable = [v for v in target_data]
				box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

				return box_loss, iou_loss, class_loss


			return delta_pred, conf_pred, class_pred
	model = Yolov2()
	weightloader = WeightLoader()
	pytorch_model = weightloader.load(pytorch_model, model, './data/pretrained/yolov2-tiny-voc.weights')
	# weightloader.load(model, self.scratch_torch, './yolov2-tiny-voc.weights')
  
_Dataset = False
if _Dataset:
	dataset = 'voc0712trainval'
	imdb_name = 'voc_2007_trainval+voc_2012_trainval'
	imdbval_name ='voc_2007_test'

	def axpy(N: int = 0., ALPHA: float = 1., X: int = 0, INCX: int = 1, Y: float =0., INCY: int = 1):
		for i in range(N):
			Y[i * INCY] += ALPHA * X[i * INCX]

	def get_dataset(datasetnames):
		names = datasetnames.split('+')
		dataset = RoiDataset(get_imdb(names[0]))
		print('load dataset {}'.format(names[0]))
		for name in names[1:]:
			tmp = RoiDataset(get_imdb(name))
			dataset += tmp
			print('load and add dataset {}'.format(name))
		return dataset

	train_dataset = get_dataset(imdb_name)

_Dataloader = _Dataset
if _Dataloader:
	train_dataloader = DataLoader(train_dataset, batch_size=64,
									shuffle=True, num_workers=2,
									collate_fn=detection_collate, drop_last=True)
	train_data_iter = iter(train_dataloader)

_Get_Next_Data = _Dataloader
if _Get_Next_Data:
	_data = next(train_data_iter)
	im_data, gt_boxes, gt_classes, num_obj = _data

	im_data     = im_data[0].unsqueeze(0)
	gt_boxes    = gt_boxes[0].unsqueeze(0)
	gt_classes  = gt_classes[0].unsqueeze(0)
	num_obj     = num_obj[0].unsqueeze(0)
 
	__data = im_data, gt_boxes, gt_classes, num_obj

	Path("Temp_Files").mkdir(parents=True, exist_ok=True)
	with open('Temp_Files/default_data.pickle','wb') as handle:
		pickle.dump(_data,handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	with open('Input_Data.pickle', 'rb') as handle:
		b = pickle.load(handle)
	im_data, gt_boxes, gt_classes, num_obj = b
	im_data, gt_boxes, gt_classes, num_obj = im_data[0].unsqueeze(0), gt_boxes[0].unsqueeze(0), gt_classes[0].unsqueeze(0), num_obj[0].unsqueeze(0)
	__data = im_data, gt_boxes, gt_classes, num_obj
	print(f"\n\nLoading data from saved file\n\nImage (im_data[0,:3,66:69,66:69]\n{im_data[0,:3,66:69,66:69]}\n\n")
	


if __name__ == '__main__':
    Fout, Fcache, loss, loss_grad, BlDout, Bgrads = pytorch_model.train(im_data, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)
