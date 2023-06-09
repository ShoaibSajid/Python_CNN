import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loss import build_target, yolo_loss
from cnn_python import *
from pathlib import Path

python_model = DeepConvNet(input_dims=(3, 416, 416),
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
	python_model = weightloader.load(python_model, model, './data/pretrained/yolov2-tiny-voc.weights')
	# weightloader.load(model, self.scratch_torch, './yolov2-tiny-voc.weights')

_Dataset = True
if _Dataset:
	from dataset.factory import get_imdb
	from dataset.roidb import RoiDataset, detection_collate
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

	Path("Input").mkdir(parents=True, exist_ok=True)
	with open('Input/Input_Data.pickle','wb') as handle:
		pickle.dump(__data,handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	with open('Input/Input_Data.pickle', 'rb') as handle:
		b = pickle.load(handle)
	im_data, gt_boxes, gt_classes, num_obj = b
	# im_data, gt_boxes, gt_classes, num_obj = im_data[0].unsqueeze(0), gt_boxes[0].unsqueeze(0), gt_classes[0].unsqueeze(0), num_obj[0].unsqueeze(0)
	__data = im_data, gt_boxes, gt_classes, num_obj
	print(f"\n\nLoading data from saved file\n\nImage (im_data[0,:3,66:69,66:69]\n{im_data[0,:3,66:69,66:69]}\n\n")
	

	# im = np.random.randn(1, 3, 416, 416)

	# box_loss, iou_loss, class_loss = scratch.loss(im_data, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)

if __name__ == '__main__':
	Fout, Fcache, loss, loss_grad, BlDout, Bgrads = python_model.train(im_data, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_obj)

