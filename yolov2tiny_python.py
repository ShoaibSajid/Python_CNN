from pathlib import Path
import pickle
import torch
from torch.utils.data import DataLoader
from cnn_python import *

python_model = DeepConvNet(input_dims=(3, 416, 416),
						num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
						max_pools=[0, 1, 2, 3, 4],
						weight_scale='kaiming',
						batchnorm=True,
						dtype=torch.float32, device='cpu')

_Load_Weights = True
if _Load_Weights:
	model = Yolov2()
	weightloader = WeightLoader()
	python_model = weightloader.load(python_model, model, './data/pretrained/yolov2-tiny-voc.weights')

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

