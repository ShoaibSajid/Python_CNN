from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config as cfg
from pathlib import Path

import numpy as np
import pickle
import math

# Zip the pickle file
import bz2file as bz2
warnings.simplefilter("ignore", UserWarning)

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a Python_ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               slowpool=True,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
    self.slowpool = slowpool
    self.num_filters = num_filters
    self.save_pickle = False
    self.save_output = False
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    # Replace "pass" statement with your code
    filter_size = 3
    conv_param = {'stride': 1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pred_filters,H_out,W_out = input_dims
    HH = filter_size
    WW =  filter_size
    # print('num_filters:', num_filters)
    for i,num_filter in enumerate(num_filters):
      H_out = int(1 + (H_out + 2 * conv_param['pad'] - HH) / conv_param['stride'])
      W_out = int(1 + (W_out + 2 * conv_param['pad'] - WW) / conv_param['stride'])
      if self.batchnorm:
        self.params['running_mean{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
        self.params['running_var{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
        self.params['gamma{}'.format(i)] =0.01*torch.randn(num_filter, device =device, dtype = dtype)
        self.params['beta{}'.format(i)] = 0.01*torch.randn(num_filter, device =device, dtype = dtype)
      if i in max_pools:
        H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
        W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
      if weight_scale == 'kaiming':
        self.params['W{}'.format(i)] = kaiming_initializer(num_filter, pred_filters, K=filter_size, relu=True, device=device,dtype=dtype)
      else:
        self.params['W{}'.format(i)] = torch.zeros(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device = device)
        self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device= device)
      pred_filters = num_filter
      # print('W_out',W_out)
      #filter_size = W_out
    i+=1
    # if weight_scale == 'kaiming':
    #     self.params['W{}'.format(i)] = kaiming_initializer(num_filter*H_out*W_out, num_classes, relu=False, device=device,dtype=dtype)
    # else:
    #     self.params['W{}'.format(i)] = torch.zeros(num_filter*H_out*W_out, num_classes, dtype=dtype,device = device)
    #     self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter*H_out*W_out, num_classes, dtype=dtype,device= device)
    # self.params['b{}'.format(i)] = torch.zeros(num_classes, dtype=dtype,device= device)
    # print(i)
    if weight_scale == 'kaiming':
        self.params['W{}'.format(i)] = kaiming_initializer(125, 1024, K=1, relu=False, device=device,dtype=dtype)
    # else:
    #     self.params['W{}'.format(i)] = torch.zeros(num_filter*H_out*W_out, num_classes, dtype=dtype,device = device)
    #     self.params['W{}'.format(i)] += weight_scale*torch.randn(num_filter*H_out*W_out, num_classes, dtype=dtype,device= device)
    self.params['b{}'.format(i)] = torch.zeros(125, dtype=dtype,device= device)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_params object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      for i, num_filter in enumerate(num_filters):
        self.bn_params[i]['running_mean'] = torch.zeros(num_filter, dtype=dtype, device=device)
        self.bn_params[i]['running_var'] = torch.zeros(num_filter, dtype=dtype, device=device)

    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 3  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    # assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg
                      
  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))



  def train(self, X, gt_boxes=None, gt_classes=None, num_boxes=None):
    if self.save_module_output:
      self.save_txt=self.save_in_dec_format
      self.save_hex=self.save_in_hex_format
    else:
      self.save_txt=False
      self.save_hex=False
    
    if self.forward_prop:
      out, cache, FOut = self.forward(X)
      if self.save_pickle:
        Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
        with open('Temp_Files/Python/Forward_Out_last_layer.pickle','wb') as handle:
          pickle.dump(out,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/Forward_cache.pickle','wb') as handle:
          pickle.dump(cache,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/Forward_Out_all_layers.pickle','wb') as handle:
          pickle.dump(FOut,handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Loading previous files for Forward Propagation.")
      with open('Temp_Files/Python/Forward_Out_last_layer.pickle', 'rb') as handle:
        out = pickle.load(handle)
        out.requires_grad = True
        out.retain_grad()
      with open('Temp_Files/Python/Forward_Out_all_layers.pickle','rb') as handle:
        FOut = pickle.load(handle)
      with open('Temp_Files/Python/Forward_cache.pickle', 'rb') as handle:
        cache = pickle.load(handle)
   
    if self.cal_loss:
      loss,   loss_grad = self.loss(out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_boxes)
      if self.save_pickle:
        with open('Temp_Files/Python/loss.pickle','wb') as handle:
          pickle.dump(loss,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/loss_gradients.pickle','wb') as handle:
          pickle.dump(loss_grad,handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Loading previous files for Loss Calculation.")
      with open('Temp_Files/Python/loss.pickle', 'rb') as handle:
        loss = pickle.load(handle)
      with open('Temp_Files/Python/loss_gradients.pickle', 'rb') as handle:
        loss_grad = pickle.load(handle)
       
    if self.backward_prop:   
      lDout, grads = self.backward(loss_grad, cache)
      if self.save_pickle:
        with open('Temp_Files/Python/Backward_lDout.pickle','wb') as handle:
          pickle.dump(lDout,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/Backward_grads.pickle','wb') as handle:
          pickle.dump(grads,handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      print("Loading previous files for Backward Propagation.")
      with open('Temp_Files/Python/Backward_lDout.pickle', 'rb') as handle:
        lDout = pickle.load(handle)
      with open('Temp_Files/Python/Backward_grads.pickle', 'rb') as handle:
        grads = pickle.load(handle)

    
    # Save output for circuit team.
    if self.save_layer_output:
      save_file(f'Python/Forward/Out_Last_Layer'              , out         , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Forward/Out_Layer'                   , FOut        , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Loss/loss'                           , loss        , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Loss/loss_grad'                      , loss_grad   , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Backward/Gradients_of_Input_Features', lDout       , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Backward/Gradients_'                 , grads       , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Input_Image'                         , X           , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      save_file(f'Python/Parameters/'                         , self.params , save_hex=self.save_in_hex_format, save_txt=self.save_in_dec_format )
      # save_file(f'Python/Forward/cache_Layer'                 , cache       , save_hex=self.save_hex )
      print('Outputs have been saved')
  
    return out, cache, loss, loss_grad, lDout, grads
  
  
  
  def forward(self, X):
    print(f'\nThis is python-based forward propagation code\n')
    y = 1
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_params in self.bn_params:
        bn_params['mode'] = mode

    scores = None
    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    slowpool_param = {'pool_height':2, 'pool_width':2, 'stride': 1}
    cache = {}
    Out={}
    self.phase='Forward'
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      X = convert_to_16(self, X)
    Out[0], cache['0'] = Python_Conv_BatchNorm_ReLU_Pool.forward(X                      , 
                                                                self.params['W0']       , 
                                                                self.params['gamma0']   , 
                                                                self.params['beta0']    , 
                                                                conv_param              , 
                                                                self.bn_params[0]       ,
                                                                pool_param              ,
                                                                layer_no= 0             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[0] = convert_to_16(self, Out[0])
    Out[1], cache['1'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[0]                 , 
                                                                self.params['W1']       , 
                                                                self.params['gamma1']   , 
                                                                self.params['beta1']    , 
                                                                conv_param              , 
                                                                self.bn_params[1]       ,
                                                                pool_param              ,
                                                                layer_no= 1             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )

    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[1] = convert_to_16(self, Out[1])
    Out[2], cache['2'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[1]                 , 
                                                                self.params['W2']       , 
                                                                self.params['gamma2']   , 
                                                                self.params['beta2']    , 
                                                                conv_param              , 
                                                                self.bn_params[2]       ,
                                                                pool_param              ,
                                                                layer_no= 2             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[2] = convert_to_16(self, Out[2])
    Out[3], cache['3'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[2]                 , 
                                                                self.params['W3']       , 
                                                                self.params['gamma3']   , 
                                                                self.params['beta3']    , 
                                                                conv_param              , 
                                                                self.bn_params[3]       ,
                                                                pool_param              ,
                                                                layer_no= 3             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[3] = convert_to_16(self, Out[3])
    Out[4], cache['4'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[3]                 , 
                                                                self.params['W4']       , 
                                                                self.params['gamma4']   , 
                                                                self.params['beta4']    , 
                                                                conv_param              , 
                                                                self.bn_params[4]       ,
                                                                pool_param              ,
                                                                layer_no= 4             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[4] = convert_to_16(self, Out[4])
    Out[5], cache['5'] = Python_Conv_BatchNorm_ReLU.forward     (Out[4]                 , 
                                                                self.params['W5']       , 
                                                                self.params['gamma5']   , 
                                                                self.params['beta5']    , 
                                                                conv_param              , 
                                                                self.bn_params[5]       ,
                                                                layer_no= 5             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    
    # save_file('Input' , Out[5], module='Pad', layer_no=6, save_hex=self.save_hex, save_txt=self.save_txt, phase=self.phase)
    # Out[60]            = F.pad                                  (Out[5] , (0, 1, 0, 1))
    # save_file('Output', Out[60], module='Pad', layer_no=6, save_hex=self.save_hex, save_txt=self.save_txt, phase=self.phase)
    
    
    # Out[61],cache['60']= Python_MaxPool.forward                 (Out[60]                , 
    #                                                             slowpool_param          ,
    #                                                             layer_no= 6            , 
    #                                                             save_txt= self.save_txt , 
    #                                                             save_hex= self.save_hex ,
    #                                                             phase   = self.phase    ,
    #                                                             )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[5] = convert_to_16(self, Out[5])
    Out[6], cache['6'] = Python_Conv_BatchNorm_ReLU.forward     (Out[61]                , 
                                                                self.params['W6']       , 
                                                                self.params['gamma6']   , 
                                                                self.params['beta6']    , 
                                                                conv_param              , 
                                                                self.bn_params[6]       ,
                                                                layer_no= 6             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[6] = convert_to_16(self, Out[6])
    Out[7], cache['7'] = Python_Conv_BatchNorm_ReLU.forward     (Out[6]                 , 
                                                                self.params['W7']       , 
                                                                self.params['gamma7']   , 
                                                                self.params['beta7']    , 
                                                                conv_param              , 
                                                                self.bn_params[7]       ,
                                                                layer_no= 7             , 
                                                                save_txt= self.save_txt , 
                                                                save_hex= self.save_hex ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[7] = convert_to_16(self, Out[7])
    conv_param['pad']  = 0
    Out[8], cache['8'] = Python_ConvB.forward                   (Out[7]                 , 
                                                                self.params['W8']       , 
                                                                self.params['b8']       , 
                                                                conv_param              ,
                                                                layer_no=8              , 
                                                                save_txt=self.save_txt  , 
                                                                save_hex=self.save_hex  ,
                                                                phase   = self.phase    ,
                                                                args = self  ,
                                                                )
    
    if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
      Out[8] = convert_to_16(self, Out[8])
    out = Out[8]
    
    print('\n\nFwd Out', out.dtype, out[out!=0],'\n\n')
    
    return out, cache, Out
  
  def loss(self, out, gt_boxes=None, gt_classes=None, num_boxes=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    print('Calculating the loss and its gradients for python model.')
    out = torch.tensor(out, requires_grad=True)

    scores = out
    bsize, _, h, w = out.shape
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

    xy_pred = torch.sigmoid(out[:, :, 0:2])
    conf_pred = torch.sigmoid(out[:, :, 4:5])
    hw_pred = torch.exp(out[:, :, 2:4])
    class_score = out[:, :, 5:]
    class_pred = F.softmax(class_score, dim=-1)
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)


    output_variable = (delta_pred, conf_pred, class_score)
    output_data = [v.data for v in output_variable]
    gt_data = (gt_boxes, gt_classes, num_boxes)
    target_data = build_target(output_data, gt_data, h, w)

    target_variable = [v for v in target_data]

    box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
    loss = box_loss + iou_loss + class_loss
    
    print(f"\nLoss = {loss}\n")
    out = scores
    out.retain_grad()
    loss.backward(retain_graph=True)
    dout = out.grad.detach()
    # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
    # dout = pickle.load(dout)
    # print('\n\n',dout.dtype, dout[dout!=0])
    print(f'\n\nLoss Gradients\n\t{dout.dtype}\n\t{dout[dout!=0][:10]}')
    
    # # Save output for circuit team and pickle for future.
    # if self.save_pickle:
    #   Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
    #   with open('Temp_Files/Python/Backward_loss_gradients.pickle','wb') as handle:
    #     pickle.dump(dout,handle, protocol=pickle.HIGHEST_PROTOCOL)
    # if self.save_output:
    #   Path("Outputs/Python/Backward/").mkdir(parents=True, exist_ok=True)
    #   save_file(f'Outputs/Python/Backward/Backward_loss_gradients.txt', dout)
    #   # save_file(f'Outputs/Python/Backward/Loss.txt', loss)
        
    if self.convert_to_fp16 and self.convert_loss_grad:
      dout = convert_to_16(self, dout)
      loss = convert_to_16(self, loss)
    return loss, dout
  
  def backward(self, dout, cache):
    grads = {}
    dOut={}
    self.phase ='Backwards'

    dOut[8], grads['W8'], grads['b8']                     = Python_ConvB.backward(                dout,  
                                                                                                  cache['8'], 
                                                                                                  layer_no=8              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,)
        
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[8]     = convert_to_16(self, dOut[8])
      grads['W8'] = convert_to_16(self, grads['W8'])
      grads['b8'] = convert_to_16(self, grads['b8'])
      
      
    # last_dout = 2 * last_dout
    # dw        = 2 * dw
    # db        = 2 * db

    # dw, db = grads['W8'], grads['b8']
    # print(f'\n\tdw8\n\t\t{dw.shape}\n\t\t{dw[dw!=0]}\n\t\t{dw.dtype}')
    # print(f'\n\tdb8\n\t\t{db.shape}\n\t\t{db[db!=0]}\n\t\t{db.dtype}')
    # print(f'\n\tlast_dout\n\t\t{dOut[8].shape}\n\t\t{dOut[8][dOut[8]!=0]}\n\t\t{dOut[8].dtype}')

    # print(f'\n\nBackward Grads Outputs')   
    # print(f"\n\t grads['W8']\n\t\t{grads['W8'].shape}\n\t\t{grads['W8'][grads['W8']!=0]}\n")


    dOut[7], grads['W7'], grads['gamma7'], grads['beta7']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                  dOut[8]                 , 
                                                                                                  cache['7']              , 
                                                                                                  layer_no=7              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[7]     = convert_to_16(self, dOut[7])
      grads['W7'] = convert_to_16(self, grads['W7'])
      grads['gamma7'] = convert_to_16(self, grads['gamma7'])
      grads['beta7'] = convert_to_16(self, grads['beta7'])


    dOut[6], grads['W6'], grads['gamma6'], grads['beta6']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                  dOut[7]                 , 
                                                                                                  cache['6']              , 
                                                                                                  layer_no=6              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[6]     = convert_to_16(self, dOut[6])
      grads['W6'] = convert_to_16(self, grads['W6'])
      grads['gamma6'] = convert_to_16(self, grads['gamma6'])
      grads['beta6'] = convert_to_16(self, grads['beta6'])


    dOut[5], grads['W5'], grads['gamma5'], grads['beta5']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                  dOut[6]                 , 
                                                                                                  cache['5']              , 
                                                                                                  layer_no=5              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[5]     = convert_to_16(self, dOut[5])
      grads['W5'] = convert_to_16(self, grads['W5'])
      grads['gamma5'] = convert_to_16(self, grads['gamma5'])
      grads['beta5'] = convert_to_16(self, grads['beta5'])


    dOut[4], grads['W4'], grads['gamma4'], grads['beta4']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                  dOut[5]                 , 
                                                                                                  cache['4']              , 
                                                                                                  layer_no=4              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[4]     = convert_to_16(self, dOut[4])
      grads['W4'] = convert_to_16(self, grads['W4'])
      grads['gamma4'] = convert_to_16(self, grads['gamma4'])
      grads['beta4'] = convert_to_16(self, grads['beta4'])


    dOut[3], grads['W3'], grads['gamma3'], grads['beta3']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                  dOut[4]                 , 
                                                                                                  cache['3']              , 
                                                                                                  layer_no=3              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[3]     = convert_to_16(self, dOut[3])
      grads['W3'] = convert_to_16(self, grads['W3'])
      grads['gamma3'] = convert_to_16(self, grads['gamma3'])
      grads['beta3'] = convert_to_16(self, grads['beta3'])


    dOut[2], grads['W2'], grads['gamma2'], grads['beta2']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                  dOut[3]                 , 
                                                                                                  cache['2']              , 
                                                                                                  layer_no=2              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[2]     = convert_to_16(self, dOut[2])
      grads['W2'] = convert_to_16(self, grads['W2'])
      grads['gamma2'] = convert_to_16(self, grads['gamma2'])
      grads['beta2'] = convert_to_16(self, grads['beta2'])


    dOut[1], grads['W1'], grads['gamma1'], grads['beta1']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                  dOut[2]                 , 
                                                                                                  cache['1']              , 
                                                                                                  layer_no=1              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[1]     = convert_to_16(self, dOut[1])
      grads['W1'] = convert_to_16(self, grads['W1'])
      grads['gamma1'] = convert_to_16(self, grads['gamma1'])
      grads['beta1'] = convert_to_16(self, grads['beta1'])


    dOut[0], grads['W0'], grads['gamma0'], grads['beta0']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                  dOut[1]                 , 
                                                                                                  cache['0']              , 
                                                                                                  layer_no=0              , 
                                                                                                  save_txt=self.save_txt  , 
                                                                                                  save_hex=self.save_hex  ,
                                                                                                  phase   = self.phase    ,
                                                                                                  args = self ,
                                                                                                )
    if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
      dOut[0]     = convert_to_16(self, dOut[0])
      grads['W0'] = convert_to_16(self, grads['W0'])
      grads['gamma0'] = convert_to_16(self, grads['gamma0'])
      grads['beta0'] = convert_to_16(self, grads['beta0'])


      
    # # Sa =     convert_to_16(self, 
    # # Sa

        
    # # Save pickle files for future use
    # if self.save_pickle:
    #   Path("Temp_Files/Python/").mkdir(parents=True, exist_ok=True)
    #   with open('Temp_Files/Python/Backward_dOut.pickle','wb') as handle:
    #     pickle.dump(dOut,handle, protocol=pickle.HIGHEST_PROTOCOL)
    #   with open('Temp_Files/Python/Backward_grads.pickle','wb') as handle:
    #     pickle.dump(grads,handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # if self.save_output:
    #   Path("Outputs/Python/Backward/").mkdir(parents=True, exist_ok=True)
    #   for _key in dOut.keys():
    #     save_file(f'Outputs/Python/Backward/dOut_Layer_{_key}.txt', dOut[_key])
    #   for _key in grads.keys():
    #     save_file(f'Outputs/Python/Backward/grads_Layer_{_key}.txt', grads[_key])
        
        
    return  dOut, grads

################################################################################
################################################################################
###############################  Functions Used  ###############################
################################################################################
################################################################################
   
class last_layer(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv9 = nn.Conv2d(1024, 125, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x):
    
    return self.conv9(x)


#################################################################################
# Ali and Sawera's FP32 to FP16 conversion functions

def convert_to_16(self, data):
  if self.conversion_method == 'Ali'    : data_16 = convert_ali(     self, data )
  if self.conversion_method == 'Sawera' : data_16 = convert_sawera(  self, data )
  return data_16

def convert_ali(self,     data):
  if self.convert_to_IEEE16:
    pass #Convert to IEEEFP16
  else:
    pass #Convert to BFP16
  return data

def convert_sawera(self,  data):
  if self.convert_to_IEEE16:
    pass #Convert to IEEEFP16
  else:
    pass #Convert to BFP16
  return data


#################################################################################
# Functions from Circuit Team

def convert_to_hex(value):
    # We will Use Single-Precision, Truncated and Rounding into Brain Floating Point
    # IEEE754 Single-Precision: Sign=1, Exponent_Bit=8, Mantissa_Bit=23
    Exponent_Bit = 8
    Mantissa_Bit = 23
    Binary_Value1 = Floating2Binary(value, Exponent_Bit, Mantissa_Bit)
    Hexadecimal_Value1 = hex(int(Binary_Value1, 2))[2:]
    # Truncating and Rounding
    Floating_Hexadecimal = Truncating_Rounding(Hexadecimal_Value1)
    if len(Floating_Hexadecimal) < 4:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal.zfill(4)
    else:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal
    return Brain_Floating_Hexadecimal

def Floating2Binary(num, Exponent_Bit, Mantissa_Bit):
    sign = ('1' if num < 0 else '0')
    num = abs(num)
    bias = math.pow(2, (Exponent_Bit - 1)) - 1
    if num == 0:
        e = 0
    else:
        e = math.floor(math.log(num, 2) + bias)

    if e > (math.pow(2, Exponent_Bit) - 2):  # overflow
        exponent = '1' * Exponent_Bit
        mantissa = '0' * Mantissa_Bit
    else:
        if e > 0:
            s = num / math.pow(2, e - bias) - 1
            exponent = bin(e)[2:].zfill(Exponent_Bit)
        else:  # submoral
            s = num / math.pow(2, (-bias + 1))
            exponent = '0' * Exponent_Bit
        # Rounding Mode By Adding 0.5 (Half-Rounding or Banker's Rounding)
        # Number is smaller or equal 0.5 is rounding down
        # Number is larger 0.5 is rounding up
        mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)) + 0.5))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
        # Non-Rounding Mode
        # mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
    Floating_Binary = sign + exponent + mantissa

    return Floating_Binary

def Binary2Floating(value, Exponent_Bit, Mantissa_Bit):
    sign = int(value[0], 2)
    if int(value[1:1 + Exponent_Bit], 2) != 0:
        exponent = int(value[1:1 + Exponent_Bit], 2) - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(value[1 + Exponent_Bit:], 2) * (math.pow(2, (-Mantissa_Bit))) + 1
    else:  # subnormal
        exponent = 1 - int('1' * (Exponent_Bit - 1), 2)
        # mantissa = int(value[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit)
        mantissa = int(value[1 + Exponent_Bit:], 2) * math.pow(2, (-Mantissa_Bit))
    Floating_Decimal = (math.pow(-1, sign)) * (math.pow(2, exponent)) * mantissa
    return Floating_Decimal

def Truncating_Rounding(Truncated_Hexadecimal):
    # Consider only the Length of Truncated_Hexadecimal only in [0:5]
    if len(Truncated_Hexadecimal) >= 5:
        # If this Truncated_Hexadecimal[4] >= 5 => Rounding Up the First 16 Bits
        if int(Truncated_Hexadecimal[4], 16) >= 5:
            Rounding_Hexadecimal = hex(int(Truncated_Hexadecimal[:4], 16) + 1)[2:]
        else:
            Rounding_Hexadecimal = Truncated_Hexadecimal[:4]
    else:
        Rounding_Hexadecimal = Truncated_Hexadecimal

    Rounding_Hexadecimal_Capitalized = Rounding_Hexadecimal.upper()

    return Rounding_Hexadecimal_Capitalized


# def convert_to_hex(value):
#   # IEEE16 Floating Point: sign=1bit, exponent=5bits, mantissa=10bits
#   Exponent_Bit = 5
#   Mantissa_Bit = 10
#   Bias = 15
#   Binary_Value1 = Floating2Binary(value, Exponent_Bit, Mantissa_Bit)
#   Hexadecimal_Value1 = hex(int(Binary_Value1, 2))[2:].upper()
  
#   # Previous condition
#   # if Hexadecimal_Value1 == '0':
#   #     Hexadecimal_Value1_ = Hexadecimal_Value1 + '000'
#   # else:
#   #     Hexadecimal_Value1_ = Hexadecimal_Value1
  
#   # New condition
#   if len(Hexadecimal_Value1) < 4: Hexadecimal_Value1 = Hexadecimal_Value1.zfill(4)
  
#   return Hexadecimal_Value1

# def Floating2Binary(num, Exponent_Bit, Mantissa_Bit):
#     # Designed By Thaising
#     # Combined Master-PhD in MSISLAB
#     sign = ('1' if num < 0 else '0')
#     num = abs(num)
#     bias = (2 ** (Exponent_Bit - 1) - 1)
#     e = (0 if num == 0 else math.floor(math.log(num, 2) + bias))
#     if e > (2 ** Exponent_Bit - 2):  # overflow
#         exponent = '1' * Exponent_Bit
#         mantissa = '0' * Mantissa_Bit
#     else:
#         if e > 0:  # normal
#             s = num / 2 ** (e - bias) - 1
#             exponent = bin(e)[2:].zfill(Exponent_Bit)
#         else:  # submoral
#             s = num / 2 ** (-bias + 1)
#             exponent = '0' * Exponent_Bit
#         mantissa = bin(int(s * (2 ** Mantissa_Bit) + 0.5))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
#     return sign + exponent + mantissa

# def Binary2Floating(s, Exponent_Bit, Mantissa_Bit):
#     # 2023/04/29: Designed By Thaising
#     # Combined Master-PhD in MSISLAB
#     neg = int(s[0], 2)
#     if int(s[1:1 + Exponent_Bit], 2) != 0:
#         exponent = int(s[1:1 + Exponent_Bit], 2) - int('1' * (Exponent_Bit - 1), 2)
#         mantissa = int(s[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit) + 1
#     else:  # subnormal
#         exponent = 1 - int('1' * (Exponent_Bit - 1), 2)
#         mantissa = int(s[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit)
#     return ((-1) ** neg) * (2 ** exponent) * mantissa


##################################################################################

def save_file(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
  
  if save_txt or save_hex:
    if type(data) is dict:
      for _key in data.keys():
        _fname = fname+f'_{_key}'
        save_file(_fname,data[_key])
    
    else:
      if module==[] and layer_no==[]: 
        Out_Path = f'Outputs/{os.path.split(fname)[0]}'
        fname = os.path.split(fname)[1]
      else:
        Out_Path = f'Outputs/By_Layer/'
        if layer_no!=[]: Out_Path+= f'Layer{layer_no}/'
        if module!=[]: Out_Path+= f'{module}/'
        if phase!=[]: Out_Path+= f'{phase}/'
        fname=fname
        
      if save_txt: filename = os.path.join(Out_Path, fname+'.txt')
      if save_hex: hexname  = os.path.join(Out_Path, fname+'_hex.txt')
      
      Path(Out_Path).mkdir(parents=True, exist_ok=True)
      
      if torch.is_tensor(data):
        try: data = data.detach()
        except: pass
        data = data.numpy()
      
      if save_txt: outfile = open(filename  , mode='w')
      if save_txt: outfile.write(f'{data.shape}\n')
      
      if save_hex: hexfile = open(hexname, mode='w')
      if save_hex: hexfile.write(f'{data.shape}\n')
      
      if len(data.shape)==0:
        if save_txt: outfile.write(f'{data}\n')
        if save_hex: hexfile.write(f'{data}\n')
        pass
      elif len(data.shape)==1:
        for x in data:
          if save_txt: outfile.write(f'{x}\n')
          if save_hex: hexfile.write(f'{convert_to_hex(x)}\n')
          pass
      else:
        w,x,y,z = data.shape
        for _i in range(w):
          for _j in range(x):
            for _k in range(y):
              for _l in range(z):
                _value = data[_i,_j,_k,_l]
                if save_txt: outfile.write(f'{_value}\n')
                if save_hex: hexfile.write(f'{convert_to_hex(_value)}\n')
                pass
                
      if save_hex: hexfile.close()  
      if save_txt: outfile.close()  
      
      if save_txt: print(f'\t\t--> Saved {filename}')
      if save_hex: print(f'\t\t--> Saved {hexname}')
  # else:
      # print(f'\n\t\t--> Saved {filename}')

def save_cache(fname,data):
  
  if type(data) is dict:
    for _key in data.keys():
      _fname = fname+f'_{_key}'
      save_file(_fname,data[_key])
  
  else:
    Path(os.path.split(fname)[0]).mkdir(parents=True, exist_ok=True)
    fname = fname+'.txt'
    
    if torch.is_tensor(data):
      try: data = data.detach()
      except: pass
      data = data.numpy()
    
    outfile = open(fname, mode='w')
    outfile.write(f'{data.shape}\n')
    
    if len(data.shape)==0:
      outfile.write(f'{data}\n')
    elif len(data.shape)==1:
      for x in data:
        outfile.write(f'{x}\n')
    else:
      w,x,y,z = data.shape
      for _i in range(w):
        for _j in range(x):
          for _k in range(y):
            for _l in range(z):
              outfile.write(f'{data[_i,_j,_k,_l]}\n')
    outfile.close()  
    
    print(f'\n\t\t--> Saved {fname}')

def build_target(output, gt_data, H, W):
    """
    Build the training target for output tensor

    Arguments:

    output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
    gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

    delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

    gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values
                       (x1, y1, x2, y2) range 0~1
    gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
    num_obj_batch -- tensor of shape (B, 1). number of objects


    Returns:
    iou_target -- tensor of shape (B, H * W * num_anchors, 1)
    iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
    box_target -- tensor of shape (B, H * W * num_anchors, 4)
    box_mask -- tensor of shape (B, H * W * num_anchors, 1)
    class_target -- tensor of shape (B, H * W * num_anchors, 1)
    class_mask -- tensor of shape (B, H * W * num_anchors, 1)

    """
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    gt_boxes_batch = gt_data[0]
    gt_classes_batch = gt_data[1]
    num_boxes_batch = gt_data[2]

    bsize = delta_pred_batch.size(0)

    num_anchors = 5  # hard code for now

    # initial the output tensor
    # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
    # what tensor is used doesn't matter
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    # get all the anchors

    anchors = torch.FloatTensor(cfg.anchors)

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    # this is very crucial because the predict output is normalized to 0~1, which is also
    # normalized by the grid width and height
    all_grid_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    if cfg.debug:
        print('all grid: ', all_grid_xywh[:12, :])
        print('all anchor: ', all_anchors_xywh[:12, :])
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # process over batches
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item()
        delta_pred = delta_pred_batch[b]
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H


        # step 1: process IoU target

        # apply delta_pred to pre-defined anchors
        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        box_pred = xywh2xxyy(box_pred)

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        ious = box_ious(box_pred, gt_boxes) # shape: (H * W * num_anchors, num_obj)
        ious = ious.view(-1, num_anchors, num_obj)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True) # shape: (H * W, num_anchors, 1)
        if cfg.debug:
            print('ious', ious)

        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            iou_mask[b][max_iou >= cfg.thresh] = 0

        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects

        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU

            gt_box_xywh = gt_boxes_xywh[t]
            gt_class = gt_classes[t]
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            cell_idx = cell_idx_y * W + cell_idx_x
            cell_idx = cell_idx.long()

            # update box_target, box_mask
            overlaps_in_cell = overlaps[cell_idx, :, t]
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box)
            if cfg.debug:
                print('assigned_grid, ', assigned_grid)
                print('gt: ', gt_box)
                print('target_t, ', target_t)
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            if cfg.debug:
                print(max_iou[cell_idx, argmax_anchor_idx, :])
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    return iou_target.view(bsize, -1, 1), \
           iou_mask.view(bsize, -1, 1), \
           box_target.view(bsize, -1, 4),\
           box_mask.view(bsize, -1, 1), \
           class_target.view(bsize, -1, 1).long(), \
           class_mask.view(bsize, -1, 1)

def yolo_loss(output, target):
    """
    Build yolo loss

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss
  
def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',

                        dtype=torch.float64):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If Python_ReLU=True, then initialize weights with a gain of 2 to account for
    a Python_ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################

    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    weight_scale = gain/(Din)
    weight = torch.zeros(Din,Dout, dtype=dtype,device = device)
    weight += weight_scale*torch.randn(Din,Dout, dtype=dtype,device= device)

  else:
    ###########################################################################
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    weight_scale = gain/(Din*K*K)
    weight = torch.zeros(Din,Dout, K,K, dtype=dtype,device = device)
    weight += weight_scale*torch.randn(Din,Dout, K,K, dtype=dtype,device= device)

  return weight

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[torch.arange(N), y]
  margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
  margins[torch.arange(N), y] = 0.
  loss = margins.sum() / N
  num_pos = (margins > 0).sum(dim=1)
  dx = torch.zeros_like(x)
  dx[margins > 0] = 1.
  dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
  dx /= N
  return loss, dx

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  shifted_logits = x - x.max(dim=1, keepdim=True).values
  Z = shifted_logits.exp().sum(dim=1, keepdim=True)
  log_probs = shifted_logits - Z.log()
  probs = log_probs.exp()
  N = x.shape[0]
  loss = (-1.0/ N) * log_probs[torch.arange(N), y].sum()
  dx = probs.clone()
  dx[torch.arange(N), y] -= 1
  dx /= N
  return loss, dx

def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious

def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box

def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box

def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas

def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box

def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors

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

def compressed_pickle(title, data):
	with bz2.BZ2File(title + ".pbz2", 'w') as f:
		pickle.dump(data, f)

def decompress_pickle(file):
	data = bz2.BZ2File(file, "rb")
	data = pickle.load(data)
	return data

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

################################################################################
################################################################################
#################   Python Implementations and Sandwich Layers  #################
################################################################################
################################################################################

# Python_Convolution without Bias
class Python_Conv(object):

  @staticmethod
  def forward(x, w, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, conv_param)
    """    
    
    save_file('Input'               , x  , module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Weights'             , w  , module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
      w = convert_to_16(args, w)
      
    out = None

    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    x = torch.nn.functional.pad(x, (pad,pad,pad,pad))
    
    out = torch.zeros((N,F,H_out,W_out),dtype =  x.dtype, device = x.device)

    for n in range(N):
      for f in range(F):
        for height in range(H_out):
          for width in range(W_out):
            out[n,f,height,width] = (x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] *w[f]).sum()

    cache = (x, w, conv_param)

    save_file('Output'              , out, module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      out = convert_to_16(args, out)
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, conv_param = cache
    save_file('Input'               , x, module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Weights'             , w, module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Loss_Gradients'      ,dout,module='Conv',layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
      w = convert_to_16(args, w)
      
    dx, dw  = None, None

    pad = conv_param['pad']
    stride = conv_param['stride']
    
    N,F,H_dout,W_dout = dout.shape
    F,C,HH,WW = w.shape
    
    dw = torch.zeros_like(w)
    dx = torch.zeros_like(x)
    
    for n in range(N):
      
      for f in range(F):
        for height in range(H_dout):
          for width in range(W_dout):
            dw[f]+= x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] * dout[n,f,height,width]
            dx[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW]+=w[f] * dout[n,f,height,width]
         
    dx = dx[:,:,1:-1,1:-1] # delete padded "pixels"
    
    save_file('Gradient_Of_Input'   , dx, module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Gradient_Of_Weights' , dw, module='Conv', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)

    if args.convert_to_fp16 and args.convert_module_IpOp:
      dx = convert_to_16(args, dx)
      dw = convert_to_16(args, dw)
    return dx, dw
      
# Python_Convolution with Bias
class Python_ConvB(object):

  @staticmethod
  def forward(x, w, b, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """    
    save_file('Input'               , x,   module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Weights'             , w,   module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Bias'                , b,   module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
      w = convert_to_16(args, w)
      b = convert_to_16(args, b)
      
    out = None

    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    x = torch.nn.functional.pad(x, (pad,pad,pad,pad))
    
    out = torch.zeros((N,F,H_out,W_out),dtype =  x.dtype, device = x.device)

    for n in range(N):
      for f in range(F):
        for height in range(H_out):
          for width in range(W_out):
            out[n,f,height,width] = (x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] *w[f]).sum() + b[f]

    cache = (x, w, b, conv_param)    

    save_file('Output'              , out, module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    if args.convert_to_fp16 and args.convert_module_IpOp:
      out = convert_to_16(args, out)
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    
    x, w, b, conv_param = cache
    save_file('Input'               , x,  module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Weights'             , w,  module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Bias'                , b,  module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    save_file('Loss_Gradients'      ,dout,module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    dx, dw, db = None, None, None

    pad = conv_param['pad']
    stride = conv_param['stride']
    N,F,H_dout,W_dout = dout.shape
    F,C,HH,WW = w.shape
    
    db = torch.zeros_like(b)
    dw = torch.zeros_like(w)
    dx = torch.zeros_like(x)
    
    for n in range(N):
      for f in range(F):
        for height in range(H_dout):
          for width in range(W_dout):
            db[f]+=dout[n,f,height,width]
            dw[f]+= x[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW] * dout[n,f,height,width]
            dx[n,:,height*stride:height*stride+HH,width*stride:width*stride+WW]+=w[f] * dout[n,f,height,width]
    if pad != 0:   
      dx = dx[:,:,1:-1,1:-1] # delete padded "pixels"

    save_file('Gradient_Of_Input'   , dx, module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Gradient_Of_Weights' , dw, module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Gradient_Of_Bias'    , db, module='ConvWB', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)

    return dx, dw, db

class Python_MaxPool(object):

  @staticmethod
  def forward(x, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    
    save_file('Input'               , x  , module='MaxPool', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    out = None

    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
      
    stride = pool_param['stride']
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    N,C,H,W = x.shape
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = torch.zeros((N,C,H_out,W_out),dtype =  x.dtype, device = x.device)
    for n in range(N):
        for height in range(H_out):
          for width in range(W_out):
            val, _ = x[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width].reshape(C,-1).max(dim = 1)
            out[n,:,height,width] = val

    cache = (x, pool_param)
    save_file('Output'              , out, module='MaxPool', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      out = convert_to_16(args, out)
      
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    
    x, pool_param = cache
    save_file('Input_Features'      , x  , module='MaxPool', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    save_file('Loss_Gradients'      ,dout, module='MaxPool', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
    dx = None

    N,C,H,W = x.shape
    stride = pool_param['stride']
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    dx = torch.zeros_like(x)
    for n in range(N):
        for height in range(H_out):
         for width in range(W_out):
            local_x  = x[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width]
            shape_local_x = local_x.shape
            reshaped_local_x = local_x.reshape(C ,-1)
            local_dw = torch.zeros_like(reshaped_local_x)
            values, indicies = reshaped_local_x.max(-1)
            local_dw[range(C),indicies] =  dout[n,:,height,width]
            dx[n,:,height*stride:height*stride+pool_height,width*stride:width*stride+pool_width] = local_dw.reshape(shape_local_x)

    save_file('Gradients_of_Input'  , dx , module='MaxPool', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      dx = convert_to_16(args, dx)
    return dx

class Python_BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_params, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_params: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    
    save_file('Input'      ,x,module='BatchNorm', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
      
    mode = bn_params['mode']
    eps = bn_params.get('eps', 1e-5)
    momentum = bn_params.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    if args.convert_to_fp16 and args.convert_module_IpOp:
      x = convert_to_16(args, x)
      gamma = convert_to_16(args, gamma)
      beta = convert_to_16(args, beta)
      running_mean = convert_to_16(args, running_mean)
      running_var = convert_to_16(args, running_var)
      
      
    out, cache = None, None
    if mode == 'train':

      #step1: calculate mean
      mu = 1./N * torch.sum(x, axis = 0)
      running_mean = momentum * running_mean + (1 - momentum) * mu

      #step2: subtract mean vector of every trainings example
      xmu = x - mu
      
      #step3: following the lower branch - calculation denominator
      sq = xmu ** 2
      
      #step4: calculate variance
      var = 1./N * torch.sum(sq, axis = 0)
      running_var = momentum * running_var + (1 - momentum) * var
      #step5: add eps for numerical stability, then sqrt
      sqrtvar = torch.sqrt(var + eps)

      #step6: invert sqrtwar
      ivar = 1./sqrtvar
    
      #step7: execute normalization
      xhat = xmu * ivar

      #step8: Nor the two transformation steps
      #print(gamma)

      gammax = gamma * xhat

      #step9
      out = gammax + beta

      cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

    elif mode == 'test':

      normolized = ((x - running_mean)/(running_var+ eps)**(1/2))
      out = normolized * gamma + beta

    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    if args.convert_to_fp16 and args.convert_module_IpOp:
      out = convert_to_16(args, out)
      running_mean = convert_to_16(args, running_mean)
      running_var = convert_to_16(args, running_var)
      
    # Store the updated running means back into bn_params
    bn_params['running_mean'] = running_mean.detach()
    bn_params['running_var'] = running_var.detach()
    
    save_file('Output'     ,out,module='BatchNorm', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
    
      
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

      
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    
    if args.convert_to_fp16 and args.convert_module_IpOp:
      dout = convert_to_16(args, dout)
      xhat   = convert_to_16(args, xhat)
      gamma  = convert_to_16(args, gamma)
      xmu  = convert_to_16(args, xmu)
      ivar   = convert_to_16(args, ivar)
      sqrtvar  = convert_to_16(args, sqrtvar)
      var  = convert_to_16(args, var)
      eps  = convert_to_16(args, eps)
    
    N,D = dout.shape

    #step9
    dbeta = torch.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable

    #step8
    dgamma = torch.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma

    #step7
    divar = torch.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

    #step6
    dsqrtvar = -1. /(sqrtvar**2) * divar

    #step5
    dvar = 0.5 * 1. /torch.sqrt(var+eps) * dsqrtvar

    #step4
    dsq = 1. /N * torch.ones((N,D),device = dout.device) * dvar

    #step3
    dxmu2 = 2 * xmu * dsq

    #step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * torch.sum(dxmu1+dxmu2, axis=0)

    #step1
    dx2 = 1. /N * torch.ones((N,D),device = dout.device) * dmu

    #step0
    dx = dx1 + dx2

    if args.convert_to_fp16 and args.convert_module_IpOp:
      dx = convert_to_16(args, dx)
      dgamma = convert_to_16(args, dgamma)
      dbeta = convert_to_16(args, dbeta)
    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None

    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    if args.convert_to_fp16 and args.convert_module_IpOp:
      dout    = convert_to_16(args, dout)
      xhat    = convert_to_16(args, xhat)
      gamma   = convert_to_16(args, gamma)
      xmu     = convert_to_16(args, xmu)
      ivar    = convert_to_16(args, ivar)
      sqrtvar = convert_to_16(args, sqrtvar)
      var     = convert_to_16(args, var)
      eps     = convert_to_16(args, eps)
      
    N,D = dout.shape
    #get the dimensions of the input/output
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(xhat * dout, dim=0)
    dx = (gamma*ivar/N) * (N*dout - xhat*dgamma - dbeta)

    if args.convert_to_fp16 and args.convert_module_IpOp:
      dx      = convert_to_16(args, dx)
      dgamma  = convert_to_16(args, dgamma)
      dbeta   = convert_to_16(args, dbeta)

    return dx, dgamma, dbeta

class Python_SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_params, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_params: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N,C,H,W = x.shape
    pre_m = x.permute(1,0,2,3).reshape(C,-1).T
    pre_m_normolized, cache= Python_BatchNorm.forward(pre_m, gamma, beta, bn_params, args=args)
    out = pre_m_normolized.T.reshape(C, N, H, W).permute(1,0,2,3)


    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N,C,H,W = dout.shape
    pre_m = dout.permute(1,0,2,3).reshape(C,-1).T
    dx, dgamma, dbeta = Python_BatchNorm.backward_alt(pre_m, cache, args=args)
    dx =dx.T.reshape(C, N, H, W).permute(1,0,2,3)

    return dx, dgamma, dbeta

class Python_Pad2d(object):
    
    @staticmethod
    def forward(x, pad_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
        save_file('Input'               , x  , module='Pad', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        
        if args.convert_to_fp16 and args.convert_module_IpOp:
          x = convert_to_16(args, x)
          
        new_shape = tuple(
            left + size + right
            for size, (left, right) in zip(x.shape, pad_param)
        )

        out = torch.zeros(new_shape)


        original_area_slice = tuple(
            slice(left, left + size)
            for size, (left, right) in zip(x.shape, pad_param)
        )
        out[original_area_slice] = x

        cache = (x, pad_param)
        
        save_file('Output'              , out, module='Pad', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)

        if args.convert_to_fp16 and args.convert_module_IpOp:
          out = convert_to_16(args, out)
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):

        dx = None
        x, pad_param = cache
        N, C, H, W = x.shape
        
        if args.convert_to_fp16 and args.convert_module_IpOp:
          x = convert_to_16(args, x)
        save_file('Input_Features'      , x  , module='Pad', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        save_file('Loss_Gradients'      ,dout, module='Pad', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)

        dx = torch.zeros_like(x)

        for n in range(N):
            for f in range(C):          
                for height in range(H):
                    for width in range(W):
                        dx[n, f, height, width] = dout[n, f, height, width]
                        
        save_file('Gradients_of_Input'  , dx , module='Pad', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)

        if args.convert_to_fp16 and args.convert_module_IpOp:
          dx = convert_to_16(args, dx)
        return dx

class Python_ReLU(object):

    @staticmethod
    def forward(x, alpha=0.1, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
        if args.convert_to_fp16 and args.convert_module_IpOp:
          x = convert_to_16(args, x)
        save_file('Input'               , x  , module='ReLU', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        
        out = None
        out = x.clone()
        out[out < 0] = out[out < 0] * alpha
        cache = x

        save_file('Output'              , out, module='ReLU', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        
        if args.convert_to_fp16 and args.convert_module_IpOp:
          out = convert_to_16(args, out)
          
        return out, cache

    @staticmethod
    def backward(dout, cache, alpha=0.1, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
          
        dx, x = None, cache
        save_file('Input_Features'      , x  , module='ReLU', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        save_file('Loss_Gradients'      ,dout, module='ReLU', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        if args.convert_to_fp16 and args.convert_module_IpOp:
          x = convert_to_16(args, x)
        

        dl = torch.ones_like(x)
        dl[x < 0] = alpha
        dx = dout * dl
            
        save_file('Gradients_of_Input'  , dx , module='ReLU', layer_no=layer_no, save_hex=save_hex, save_txt=save_txt, phase=phase)
        if args.convert_to_fp16 and args.convert_module_IpOp:
          dx = convert_to_16(args, dx)
        
        return dx







class Python_Conv_ReLU(object):

  @staticmethod
  def forward(x, w, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A convenience layer that performs a convolution followed by a Python_ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the Python_ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache   = Python_Conv.forward(
                                            x, 
                                            w, 
                                            conv_param, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,)
    out, relu_cache = Python_ReLU.forward(
                                            a, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )
    cache = (conv_cache, relu_cache)
    print(f'{layer_no}',end=',')
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da      = Python_ReLU.backward(
                                            dout, 
                                            relu_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )
    dx, dw  = Python_Conv.backward(
                                            da, 
                                            conv_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )
    print(f'{layer_no}',end=',')
    return dx, dw

class Python_Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, conv_param, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    A convenience layer that performs a convolution, a Python_ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache   = Python_Conv.forward(
                                            x, 
                                            w, 
                                            conv_param, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    s, relu_cache   = Python_ReLU.forward(
                                            a, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    out, pool_cache = Python_MaxPool.forward(
                                            s, 
                                            pool_param, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    cache = (conv_cache, relu_cache, pool_cache)
    print(f'{layer_no}',end=',')
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds            = Python_MaxPool.backward(
                                            dout, 
                                            pool_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    da            = Python_ReLU.backward(
                                            ds, 
                                            relu_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    dx, dw        = Python_Conv.backward(
                                            da, 
                                            conv_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    print(f'{layer_no}',end=',')
    return dx, dw

class Python_Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_params, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    a, conv_cache   = Python_Conv.forward(
                                            x, 
                                            w, 
                                            conv_param, 
                                            layer_no=layer_no,
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,)

    an, bn_cache    = Python_SpatialBatchNorm.forward(
                                            a, 
                                            gamma, 
                                            beta, 
                                            bn_params, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,)

    out, relu_cache = Python_ReLU.forward(
                                            an, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,)

    cache = (conv_cache, bn_cache, relu_cache)
    print(f'{layer_no}',end=',')
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    conv_cache, bn_cache, relu_cache = cache
    
    dan               = Python_ReLU.backward(
                                            dout, 
                                            relu_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    da, dgamma, dbeta = Python_SpatialBatchNorm.backward(
                                            dan, 
                                            bn_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    dx, dw            = Python_Conv.backward(
                                            da, 
                                            conv_cache, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    print(f'{layer_no}',end=',')
    return dx, dw, dgamma, dbeta

class Python_Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_params, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    # Load in FP16 --> FP 32
    a,    conv_cache  = Python_Conv.forward(
                                            x, 
                                            w, 
                                            conv_param, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )
    # Convert 32-> 16 --> Save in FP16
    
    # Load in FP16 --> FP 32
    an,   bn_cache    = Python_SpatialBatchNorm.forward(
                                            a, 
                                            gamma, 
                                            beta, 
                                            bn_params, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    s,    relu_cache  = Python_ReLU.forward(
                                            an, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    out,  pool_cache  = Python_MaxPool.forward(
                                            s, 
                                            pool_param, 
                                            layer_no=layer_no, 
                                            save_txt=save_txt, 
                                            save_hex=save_hex, 
                                            phase=phase,
                                            args = args,
                                            )

    
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    print(f'{layer_no}',end=',')
    return out, cache

  @staticmethod
  def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[], args=None):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    
    ds                = Python_MaxPool.backward(
                                                dout, 
                                                pool_cache, 
                                                layer_no=layer_no, 
                                                save_txt=save_txt, 
                                                save_hex=save_hex, 
                                                phase=phase,
                                                args = args,
                                                )

    dan               = Python_ReLU.backward(
                                                ds, 
                                                relu_cache, 
                                                layer_no=layer_no, 
                                                save_txt=save_txt, 
                                                save_hex=save_hex, 
                                                phase=phase,
                                                args = args,
                                                )

    da, dgamma, dbeta = Python_SpatialBatchNorm.backward(
                                                dan, 
                                                bn_cache, 
                                                layer_no=layer_no, 
                                                save_txt=save_txt, 
                                                save_hex=save_hex, 
                                                phase=phase,
                                                args = args,
                                                )

    dx, dw            = Python_Conv.backward(
                                                da, 
                                                conv_cache, 
                                                layer_no=layer_no, 
                                                save_txt=save_txt, 
                                                save_hex=save_hex, 
                                                phase=phase,
                                                args = args,
                                                )

    
    print(f'{layer_no}',end=',')
    return dx, dw, dgamma, dbeta
