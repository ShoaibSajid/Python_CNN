import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightLoader(object):
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None
        self.b = 'b'
        self.g = 'g'
        self.rm = 'rm'
        self.rv = 'rv'
        
    def load_conv_bn(self, model, conv_model, bn_model):
        
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.bias.data.shape == model.params['beta0'].shape:
            model.params['beta0'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == model.params['beta1'].shape:
            model.params['beta1'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == model.params['beta2'].shape:
            model.params['beta2'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == model.params['beta3'].shape:
            model.params['beta3'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == model.params['beta4'].shape:
            model.params['beta4'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == model.params['beta5'].shape:
            model.params['beta5'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif (bn_model.bias.data.shape == model.params['beta6'].shape) and self.b == "b":
            model.params['beta6'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.bias.data))
            self.b = 'bb'
        elif (model.params['beta7'].shape == bn_model.bias.data.shape) and self.b == "bb":
            model.params['beta7'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.bias.data))

        self.start = self.start + num_b

        

        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))


        if bn_model.weight.data.shape == model.params['gamma0'].shape:
            model.params['gamma0'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == model.params['gamma1'].shape:
            model.params['gamma1'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == model.params['gamma2'].shape:
            model.params['gamma2'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == model.params['gamma3'].shape:
            model.params['gamma3'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == model.params['gamma4'].shape:
            model.params['gamma4'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == model.params['gamma5'].shape:
            model.params['gamma5'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif (bn_model.weight.shape == model.params['gamma6'].shape) and self.g == "g":
            model.params['gamma6'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.weight.data))
            self.g = 'gg'
        elif (model.params['gamma7'].shape == bn_model.weight.data.shape) and self.g == "gg":
            model.params['gamma7'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.weight.data))

        self.start = self.start + num_b

        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_mean.data.shape == model.bn_params[0]['running_mean'].shape:
            model.bn_params[0]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[1]['running_mean'].shape:
            model.bn_params[1]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[2]['running_mean'].shape:
            model.bn_params[2]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[3]['running_mean'].shape:
            model.bn_params[3]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[4]['running_mean'].shape:
            model.bn_params[4]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[5]['running_mean'].shape:
            model.bn_params[5]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == model.bn_params[6]['running_mean'].shape and self.rm == "rm":
            model.bn_params[6]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
            self.rm = "rmrm"
        elif bn_model.running_mean.data.shape == model.bn_params[7]['running_mean'].shape and self.rm == "rmrm":
            model.bn_params[7]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_mean.data))

        self.start = self.start + num_b

        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_var.data.shape == model.bn_params[0]['running_var'].shape:
            model.bn_params[0]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[1]['running_var'].shape:
            model.bn_params[1]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[2]['running_var'].shape:
            model.bn_params[2]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[3]['running_var'].shape:
            model.bn_params[3]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[4]['running_var'].shape:
            model.bn_params[4]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[5]['running_var'].shape:
            model.bn_params[5]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == model.bn_params[6]['running_var'].shape and self.rv == "rv":
            model.bn_params[6]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_var.data))
            self.rv = "rvrv"
        elif bn_model.running_var.data.shape == model.bn_params[7]['running_var'].shape and self.rv == "rvrv":
            model.bn_params[7]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_var.data))
            
        self.start = self.start + num_b

        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        
        
        if conv_model.weight.data.shape == (16, 3, 3, 3):
            model.params['W0'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(0), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (32, 16, 3, 3):
            model.params['W1'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(1), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (64, 32, 3, 3):
            model.params['W2'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(2), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (128, 64, 3, 3):
            model.params['W3'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(3), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (256, 128, 3, 3):
            model.params['W4'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(4), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (512, 256, 3, 3):
            model.params['W5'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(5), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 512, 3, 3):            
            model.params['W6'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(6), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 1024, 3, 3):
            model.params['W7'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(7), mode='w') as f:
                f.write(str(conv_model.weight.data))
        self.start = self.start + num_w

    def load_conv(self, model, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        model.params['b8'] = conv_model.bias.data
        with open('./weight_parameter/bias/{}'.format(7), mode='w') as f:
            f.write(str(conv_model.bias.data))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        model.params['W8'] = conv_model.weight.data
        with open('./weight_parameter/conv_param/w/{}'.format(8), mode='w') as f:
            f.write(str(conv_model.weight.data))
        self.start = self.start + num_w

    def dfs(self, org_model=[], scratch_model=[]):
        children = list(org_model.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(org_model=c, scratch_model=scratch_model)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(scratch_model, c)
                else:
                    self.load_conv_bn(scratch_model, c, children[i + 1])

    def load(self, load_from=[], load_to=[], weights_file=[]):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(load_from, load_to)
        # make sure the loaded weight is right
    
        assert size == self.start