# %%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov2tiny import Yolov2
import eecs598
from cnn_torch import DeepConvNetTorch, FastConv, FastConvWB, Conv_BatchNorm_ReLU, Conv_BatchNorm_ReLU_Pool, Conv_ReLU
from cnn_scratch import DeepConvNet

# %%
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

        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.bias.data.shape == scratch.params['beta0'].shape:
            scratch.params['beta0'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == scratch.params['beta1'].shape:
            scratch.params['beta1'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == scratch.params['beta2'].shape:
            scratch.params['beta2'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == scratch.params['beta3'].shape:
            scratch.params['beta3'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == scratch.params['beta4'].shape:
            scratch.params['beta4'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == scratch.params['beta5'].shape:
            scratch.params['beta5'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif (bn_model.bias.data.shape == scratch.params['beta6'].shape) and self.b == "b":
            scratch.params['beta6'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.bias.data))
            self.b = 'bb'
        elif (scratch.params['beta7'].shape == bn_model.bias.data.shape) and self.b == "bb":
            scratch.params['beta7'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.bias.data))

        self.start = self.start + num_b

        

        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))


        if bn_model.weight.data.shape == scratch.params['gamma0'].shape:
            scratch.params['gamma0'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == scratch.params['gamma1'].shape:
            scratch.params['gamma1'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == scratch.params['gamma2'].shape:
            scratch.params['gamma2'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == scratch.params['gamma3'].shape:
            scratch.params['gamma3'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == scratch.params['gamma4'].shape:
            scratch.params['gamma4'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == scratch.params['gamma5'].shape:
            scratch.params['gamma5'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif (bn_model.weight.shape == scratch.params['gamma6'].shape) and self.g == "g":
            scratch.params['gamma6'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.weight.data))
            self.g = 'gg'
        elif (scratch.params['gamma7'].shape == bn_model.weight.data.shape) and self.g == "gg":
            scratch.params['gamma7'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.weight.data))

        self.start = self.start + num_b

        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_mean.data.shape == scratch.bn_params[0]['running_mean'].shape:
            scratch.bn_params[0]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[1]['running_mean'].shape:
            scratch.bn_params[1]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[2]['running_mean'].shape:
            scratch.bn_params[2]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[3]['running_mean'].shape:
            scratch.bn_params[3]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[4]['running_mean'].shape:
            scratch.bn_params[4]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[5]['running_mean'].shape:
            scratch.bn_params[5]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == scratch.bn_params[6]['running_mean'].shape and self.rm == "rm":
            scratch.bn_params[6]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
            self.rm = "rmrm"
        elif bn_model.running_mean.data.shape == scratch.bn_params[7]['running_mean'].shape and self.rm == "rmrm":
            scratch.bn_params[7]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_mean.data))

        self.start = self.start + num_b

        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_var.data.shape == scratch.bn_params[0]['running_var'].shape:
            scratch.bn_params[0]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[1]['running_var'].shape:
            scratch.bn_params[1]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[2]['running_var'].shape:
            scratch.bn_params[2]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[3]['running_var'].shape:
            scratch.bn_params[3]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[4]['running_var'].shape:
            scratch.bn_params[4]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[5]['running_var'].shape:
            scratch.bn_params[5]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == scratch.bn_params[6]['running_var'].shape and self.rv == "rv":
            scratch.bn_params[6]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_var.data))
            self.rv = "rvrv"
        elif bn_model.running_var.data.shape == scratch.bn_params[7]['running_var'].shape and self.rv == "rvrv":
            scratch.bn_params[7]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_var.data))
            
        self.start = self.start + num_b

        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        
        
        if conv_model.weight.data.shape == (16, 3, 3, 3):
            scratch.params['W0'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(0), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (32, 16, 3, 3):
            scratch.params['W1'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(1), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (64, 32, 3, 3):
            scratch.params['W2'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(2), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (128, 64, 3, 3):
            scratch.params['W3'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(3), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (256, 128, 3, 3):
            scratch.params['W4'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(4), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (512, 256, 3, 3):
            scratch.params['W5'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(5), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 512, 3, 3):            
            scratch.params['W6'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(6), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 1024, 3, 3):
            scratch.params['W7'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(7), mode='w') as f:
                f.write(str(conv_model.weight.data))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        scratch.params['b8'] = conv_model.bias.data
        with open('./weight_parameter/bias/{}'.format(7), mode='w') as f:
            f.write(str(conv_model.bias.data))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        scratch.params['W8'] = conv_model.weight.data
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

    def load(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)
        # make sure the loaded weight is right
    
        assert size == self.start









# %%
scratch = DeepConvNet(input_dims=(3, 416, 416),
                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                        max_pools=[0, 1, 2, 3, 4],
                        weight_scale='kaiming',
                        batchnorm=True,
                        dtype=torch.float32, device='cpu')

# %%
scratch = DeepConvNetTorch(input_dims=(3, 416, 416),
                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                        max_pools=[0, 1, 2, 3, 4],
                        weight_scale='kaiming',
                        batchnorm=True,
                        dtype=torch.float32, device='cpu')

# %%
model = Yolov2()

# %%
weightloader = WeightLoader()
weightloader.load(model, './yolov2-tiny-voc.weights')
# weightloader.load(model, scratch_torch, './yolov2-tiny-voc.weights')

# %%
x = torch.randn(1, 3, 416, 416)

# %%
output_scratch = scratch.loss(x)
print(output_scratch)

# %%
output_torch = scratch.loss(x)
print(output_torch)

# %%
eecs598.grad.rel_error(output_scratch[0], output_torch[0])

# %%
eecs598.grad.rel_error(output_scratch[1], output_torch[1])

# %%
eecs598.grad.rel_error(output_scratch[2], output_torch[2])

# %%



