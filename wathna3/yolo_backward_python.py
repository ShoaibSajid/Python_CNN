from pathlib import Path
import pickle
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch
from config import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform
import torch.nn.functional as F

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

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
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

    weight_scale = gain/(Din)
    weight = torch.zeros(Din,Dout, dtype=dtype,device = device)
    weight += weight_scale*torch.randn(Din,Dout, dtype=dtype,device= device)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:

    weight_scale = gain/(Din*K*K)
    weight = torch.zeros(Din,Dout, K,K, dtype=dtype,device = device)
    weight += weight_scale*torch.randn(Din,Dout, K,K, dtype=dtype,device= device)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight

class last_layer(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv9 = nn.Conv2d(1024, 125, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x):
    
    return self.conv9(x)

class DeepConvNet(object): # Python based CNN Implementation
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
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
    self.p = {} # Parameters
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
    self.slowpool = slowpool
    self.num_filters = num_filters
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.p.                                                  #
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
        self.p['running_mean{}'.format(i)]  = torch.zeros(num_filter, dtype=dtype, device=device)
        self.p['running_var{}'.format(i)]   = torch.zeros(num_filter, dtype=dtype, device=device)
        self.p['gamma{}'.format(i)]         = 0.01*torch.randn(num_filter, device =device, dtype = dtype)
        self.p['beta{}'.format(i)]          = 0.01*torch.randn(num_filter, device =device, dtype = dtype)
        
      if i in max_pools:
        H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
        W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
      
      if weight_scale == 'kaiming':
        self.p['W{}'.format(i)]  = kaiming_initializer(num_filter, pred_filters, K=filter_size, relu=True, device=device,dtype=dtype)
      else:
        self.p['W{}'.format(i)]  = torch.zeros(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device = device)
        self.p['W{}'.format(i)] += weight_scale*torch.randn(num_filter,pred_filters, filter_size,filter_size, dtype=dtype,device= device)
      
      pred_filters = num_filter
      
      # print('W_out',W_out)
      # filter_size = W_out
    i+=1
    if weight_scale == 'kaiming':
        self.p['W{}'.format(i)] = kaiming_initializer(125, 1024, K=1, relu=False, device=device,dtype=dtype)
    self.p['b{}'.format(i)] = torch.zeros(125, dtype=dtype,device= device)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
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
    msg = 'self.p has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.p), num_params)
    # assert len(self.p) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, p in self.p.items():
      msg = 'p "%s" has device %r; should be %r' % (k, p.device, device)
      assert p.device == torch.device(device), msg
      msg = 'p "%s" has dtype %r; should be %r' % (k, p.dtype, dtype)
      assert p.dtype == dtype, msg

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    for i,num_filter in enumerate(num_filters):
      self.p[f'conv{i}'] = {'stride': 1, 'pad': (filter_size - 1) // 2}
    self.p[f'conv{8}'] = {'stride': 1, 'pad': (filter_size - 1) // 2}
    self.p[f'conv{8}']['pad']   = 0
    

  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.p,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.p = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.p:
      self.p[p] = self.p[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))

  def forward(self, X=[], save_output=False, save_pickle=False, Forward_Prop=True, is_training=True):
    print(f"\nThis code is using Python based code.")
    X     = X.to(self.dtype)
    cache = {}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param      = {'pool_height': 2, 'pool_width': 2,   'stride': 2}
    slowpool_param  = {'pool_height':2,  'pool_width':2,    'stride': 1}
    
    # Set train/test mode for batchnorm params since they behave differently during training and testing.
    mode = 'train' if is_training else 'test'
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    
    
    # Forward Propagation simplified code ---- ON CHIP
    if Forward_Prop:
      print(f"\tForward Propagation")
      Out={}
      Out[0] , cache['0']  = Conv_BatchNorm_ReLU_Pool.forward(X      , self.p['W0'], self.p['gamma0'],self.p['beta0'],self.p[f'conv0'],self.bn_params[0],pool_param)
      Out[1] , cache['1']  = Conv_BatchNorm_ReLU_Pool.forward(Out[0] , self.p['W1'], self.p['gamma1'],self.p['beta1'],self.p[f'conv1'],self.bn_params[1],pool_param)
      Out[2] , cache['2']  = Conv_BatchNorm_ReLU_Pool.forward(Out[1] , self.p['W2'], self.p['gamma2'],self.p['beta2'],self.p[f'conv2'],self.bn_params[2],pool_param)
      Out[3] , cache['3']  = Conv_BatchNorm_ReLU_Pool.forward(Out[2] , self.p['W3'], self.p['gamma3'],self.p['beta3'],self.p[f'conv3'],self.bn_params[3],pool_param)
      Out[4] , cache['4']  = Conv_BatchNorm_ReLU_Pool.forward(Out[3] , self.p['W4'], self.p['gamma4'],self.p['beta4'],self.p[f'conv4'],self.bn_params[4],pool_param)
      Out[5] , cache['5']  = Conv_BatchNorm_ReLU.forward     (Out[4] , self.p['W5'], self.p['gamma5'],self.p['beta5'],self.p[f'conv5'],self.bn_params[5]) 
      Out[60]              = F.pad                           (Out[5] , (0, 1, 0, 1))
      Out[61], cache['61'] = MaxPool.forward                 (Out[60], slowpool_param)
      Out[6] , cache['6']  = Conv_BatchNorm_ReLU.forward     (Out[61], self.p['W6'], self.p['gamma6'],self.p['beta6'],self.p[f'conv6'],self.bn_params[6]) 
      Out[7] , cache['7']  = Conv_BatchNorm_ReLU.forward     (Out[6] , self.p['W7'], self.p['gamma7'],self.p['beta7'],self.p[f'conv7'],self.bn_params[7]) 
      Out[8] , cache['8']  = ConvB.forward              (Out[7] , self.p['W8'], self.p['b8']                    ,self.p[f'conv8'])
      out = Out[8] 
          
      # Save pickle for future loading 
      if save_pickle:
        Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
        with open('Temp_Files/Python/Forward_Out.pickle','wb') as handle:
          pickle.dump(Out,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/Forward_cache.pickle','wb') as handle:
          pickle.dump(cache,handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      # Load pickle from old files
      Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
      print(f"\tForward Propagation - Load saved file.")
      with open('Temp_Files/Python/Forward_Out.pickle','rb') as handle:
        Out   = pickle.load(handle) 
        out = Out[8]
        out.requires_grad = True
        out.retain_grad()
      with open('Temp_Files/Python/Forward_cache.pickle','rb') as handle:
        cache = pickle.load(handle)   

    # Save output for circuit team.
    if save_output:
      Path("Outputs/Python/Forward/").mkdir(parents=True, exist_ok=True)
      compare = open(f'Outputs/Python/Forward/Comparison', mode='w')
      for _key in Out.keys():
        with open(f'Outputs/Python/Forward/Layer_{_key}', mode='w') as f:
          f.write(str(Out[_key]))
          compare.write(f'\n\nLayer {_key} - Forward')
          compare.write(str(Out[_key][0,2,8:13,8:13]))
    
    # Return the output of forward propagation
    return Out, cache
    
  def loss(self, gt_boxes=None, gt_classes=None, num_boxes=None, Compute_Loss=True, save_output=False, save_pickle=False):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    
    #  Compute_Loss_On_system
    if Compute_Loss:
      print(f"\tLoss Calculation done on system.")
      out.requires_grad = True
      out.retain_grad()
      # Reshape the output 
      scores          = out
      bsize, _, h, w  = out.shape
      out             = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

      # Segregate the output into [x,y,w,h], confidence, classes
      # BBox
      xy_pred     = torch.sigmoid (out[:, :, 0:2])
      hw_pred     = torch.exp     (out[:, :, 2:4])
      bbox_pred   = torch.cat([xy_pred, hw_pred], dim=-1)
      # Confidence
      conf_pred   = torch.sigmoid (out[:, :, 4:5])
      # Class
      class_score = out[:, :, 5:]
      class_pred  = F.softmax(class_score, dim=-1)

      # Reshape into output and target variable for yolo_loss computation
      # output_variable: tuple[Tensor, Tensor, Any] = (torch.cat([torch.sigmoid(out[:, :, 0:2]), torch.exp(out[:, :, 2:4])], dim=-1), torch.sigmoid(out[:, :, 4:5]), out[:, :, 5:])
      # Predicted / Output Data
      output_variable = (bbox_pred, conf_pred, class_score)
      output_data     = [v.data for v in output_variable]
      # GT Data
      gt_data         = (gt_boxes, gt_classes, num_boxes)
      # Target data for comparison
      target_data     = build_target(output_data, gt_data, h, w)
      target_variable = [v for v in target_data]

      box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
      loss = box_loss + iou_loss + class_loss
      # loss, dout = yolo_loss(output_variable, target_variable)
      
      # Calculate loss gradients
      grads = {}
      out = scores
      out.requires_grad = True
      loss.backward(retain_graph=True)
      dout = out.grad
        
      # Save output for circuit team and pickle for future.
      if save_pickle or save_output: Path("Outputs/Python/Backward/").mkdir(parents=True, exist_ok=True)
      if save_pickle:
        with open('Temp_Files/Python/Backward_loss_gradients.pickle','wb') as handle:
          pickle.dump(dout,handle, protocol=pickle.HIGHEST_PROTOCOL)
      if save_output:
        with open(f'Outputs/Python/Backward/Backward_loss_gradients', mode='w') as f:
          f.write(str(dout))
      
    # System will return dout / loss derivatives
    return dout, class_pred
    
  def backward(self, loss_gradients=[], cache=[], Backward_prop=True, save_output=False, save_pickle=False):
    dout = loss_gradients
    
    # Backward Propagation ---- ON CHIP
    if Backward_prop: 
      print(f"\tBackward Propagation")      
      dOut={}       
      dOut[8], grads['W8'], grads['b8']                     = ConvB.backward                   (dout,  cache['8']) 
      dOut[7], grads['W7'], grads['gamma7'], grads['beta7'] = Conv_BatchNorm_ReLU.backward     (dOut[8], cache['7'])
      dOut[6], grads['W6'], grads['gamma6'], grads['beta6'] = Conv_BatchNorm_ReLU.backward     (dOut[7], cache['6'])
      dOut[5], grads['W5'], grads['gamma5'], grads['beta5'] = Conv_BatchNorm_ReLU.backward     (dOut[6], cache['5'])
      dOut[4], grads['W4'], grads['gamma4'], grads['beta4'] = Conv_BatchNorm_ReLU_Pool.backward(dOut[5], cache['4'])
      dOut[3], grads['W3'], grads['gamma3'], grads['beta3'] = Conv_BatchNorm_ReLU_Pool.backward(dOut[4], cache['3'])
      dOut[2], grads['W2'], grads['gamma2'], grads['beta2'] = Conv_BatchNorm_ReLU_Pool.backward(dOut[3], cache['2'])
      dOut[1], grads['W1'], grads['gamma1'], grads['beta1'] = Conv_BatchNorm_ReLU_Pool.backward(dOut[2], cache['1'])
      dOut[0], grads['W0'], grads['gamma0'], grads['beta0'] = Conv_BatchNorm_ReLU_Pool.backward(dOut[1], cache['0'])

      # Save pickle files for future use
      if save_pickle:
        Path("Temp_Files/Python/").mkdir(parents=True, exist_ok=True)
        with open('Temp_Files/Python/Backward_dOut.pickle','wb') as handle:
          pickle.dump(dOut,handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Temp_Files/Python/Backward_grads.pickle','wb') as handle:
          pickle.dump(grads,handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      # Load data from old pickle file
      print(f"\tBackward Propagation - Load from file")      
      with open('Temp_Files/Python/Backward_dOut.pickle','rb') as handle:
        dOut   = pickle.load(handle) 
      with open('Temp_Files/Python/Backward_grads.pickle','rb') as handle:
        grads  = pickle.load(handle)  
    
    if save_output:
      for _key in dOut.keys():
        with open(f'Outputs/Python/Backward/Layer_{_key}', mode='w') as f:
          f.write(str(dOut[_key]))
          try:
            compare.write(f'\n\nLayer {_key} - Backward')
            compare.write(str(dOut[_key][0,2,8:13,8:13]))
          except:
            compare = open(f'Outputs/Python/Forward/Comparison', mode='w')
            compare.write(f'\n\nLayer {_key} - Backward')
            compare.write(str(dOut[_key][0,2,8:13,8:13]))
                  
    return  dOut, grads

  def train(self, X, gt_boxes=None, gt_classes=None, num_boxes=None):
    """
    Perform one iteration of the training
    """
    Forward_Prop = False
    Compute_Loss = False
    Backward_prop = True
    
    is_training=True
    
    save_output = False
    save_pickle = False
    
    Forward_Out, cache   = self.forward (X=X, 
                                         Forward_Prop=Forward_Prop, 
                                         is_training=is_training, 
                                         save_output=save_output, 
                                         save_pickle=save_pickle)
    
    loss_gradients, classes = self.loss (gt_boxes=gt_boxes, 
                                         gt_classes=gt_classes, 
                                         num_boxes=num_boxes, 
                                         Compute_Loss=Compute_Loss, 
                                         save_output=save_output, 
                                         save_pickle=save_pickle)
    
    dOut, grads          = self.backward(loss_gradients=loss_gradients, 
                                         cache=cache, 
                                         Backward_prop=Backward_prop, 
                                         save_output=save_output, 
                                         save_pickle=save_pickle)
    
    print('Iteration complete.')
    print(' ')

################################################################################
################################################################################
#########################   Python Implementations  ############################
################################################################################
################################################################################


# Convolution with Bias
class ConvB(object):

  @staticmethod
  def forward(x, w, b, conv_param, _debug=False):
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
    return out, cache

  @staticmethod
  def backward(dout, cache):
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
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
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
      dx = dx[:,:,1:-1,1:-1] #delete padded "pixels"
    print(dx.shape)   

    return dx, dw, db

# Convolution without Bias
class Conv(object):

  @staticmethod
  def forward(x, w, conv_param, _debug=False):
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
    return out, cache

  @staticmethod
  def backward(dout, cache):
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
    dx, dw  = None, None

    x, w, conv_param = cache
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
         
    dx = dx[:,:,1:-1,1:-1] #delete padded "pixels"
    print(dx.shape)   

    return dx, dw
 
class ReLU(object):

    @staticmethod
    def forward(x, alpha=0.1):

        out = None
        out = x.clone()
        out[out < 0] = out[out < 0] * alpha
        cache = x

        return out, cache

    @staticmethod
    def backward(dout, cache, alpha=0.1):

        dx, x = None, cache

        dl = torch.ones_like(x)
        dl[x < 0] = alpha
        dx = dout * dl

        return dx
    
class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
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
    out = None

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
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
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

    return dx

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = Conv.forward(x, w, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw = Conv.backward(da, conv_cache)
    return dx, dw

class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = Conv.forward(x, w, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = MaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = MaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw = Conv.backward(da, conv_cache)
    return dx, dw

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
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
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

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

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
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

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
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
    N,D = dout.shape
    #get the dimensions of the input/output
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(xhat * dout, dim=0)
    dx = (gamma*ivar/N) * (N*dout - xhat*dgamma - dbeta)


    return dx, dgamma, dbeta

class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
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
    pre_m_normolized, cache= BatchNorm.forward(pre_m, gamma, beta, bn_param)
    out = pre_m_normolized.T.reshape(C, N, H, W).permute(1,0,2,3)


    return out, cache

  @staticmethod
  def backward(dout, cache):
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
    dx, dgamma, dbeta = BatchNorm.backward_alt(pre_m, cache)
    dx =dx.T.reshape(C, N, H, W).permute(1,0,2,3)

    return dx, dgamma, dbeta

class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_param):
    a, conv_cache = Conv.forward(x, w, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw = Conv.backward(da, conv_cache)
    return dx, dw, dgamma, dbeta

class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = Conv.forward(x, w, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = MaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = MaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw = Conv.backward(da, conv_cache)
    return dx, dw, dgamma, dbeta
