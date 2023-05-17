import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov2tiny import Yolov2
import eecs598
from cnn_torch import DeepConvNetTorch, FastConv, FastConvWB, Conv_BatchNorm_ReLU, Conv_BatchNorm_ReLU_Pool, Conv_ReLU
from cnn_scratch import DeepConvNet
from weight_loader_2 import WeightLoader as WeightLoader_new


# Initialize Random Values
torch.manual_seed(0)
x = torch.randn(1, 3, 416, 416)
x[0][0][0][0]




# Initialize Networks
scratch_python_model = DeepConvNet(input_dims=(3, 416, 416),
                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                        max_pools=[0, 1, 2, 3, 4],
                        weight_scale='kaiming',
                        batchnorm=True,
                        dtype=torch.float32, device='cpu')

scratch_torch_model  = DeepConvNetTorch(input_dims=(3, 416, 416),
                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                        max_pools=[0, 1, 2, 3, 4],
                        weight_scale='kaiming',
                        batchnorm=True,
                        dtype=torch.float32, device='cpu')
model = Yolov2()




# Model weights before loading from weights file
print(scratch_python_model.params['W3'][0,0,0].detach().cpu().numpy())
print(scratch_torch_model.params['W3'][0,0,0].detach().cpu().numpy())



# Load weights from file
weightloader_new = WeightLoader_new()
scratch_python = weightloader_new.load(load_from=model, load_to=scratch_python_model, weights_file='./yolov2-tiny-voc.weights')
scratch_torch  = weightloader_new.load(load_from=model, load_to=scratch_torch_model , weights_file='./yolov2-tiny-voc.weights')



# Model weights after loading from weights file
print(scratch_python.params['W3'][0,0,0].detach().cpu().numpy())
print(scratch_torch.params['W3'][0,0,0].detach().cpu().numpy())



# Run forward propagation of network - Torch
output_torch  = scratch_torch.loss(x)
print()
print("Outputs from scratch model made using Torch")
print(f"\t{output_torch[0][:,200,:].detach().cpu().numpy()}")
print(f"\t{output_torch[0][0,200,1].detach().cpu().numpy()}")
print()



# Run forward propagation of network - Python
output_python = scratch_python.loss(x)
print("Outputs from scratch model made using Python")
print(f"\t{output_python[0][:,200,:].detach().cpu().numpy()}")
print(f"\t{output_python[0][0,200,1].detach().cpu().numpy()}")
print()


print(f"Diff in a random value:\n\t{output_python[0][0,200,1].detach().cpu().numpy() - output_torch[0][0,200,1].detach().cpu().numpy()}")
print('\nOverall errors')
print(f"\tError in BBox:           {eecs598.grad.rel_error(output_python[0], output_torch[0])}")
print(f"\tError in Feature Map:    {eecs598.grad.rel_error(output_python[1], output_torch[1])}")
print(f"\tError in Classification: {eecs598.grad.rel_error(output_python[2], output_torch[2])}")