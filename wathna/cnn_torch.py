import torch
import random
import torch.nn.functional as F



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

class Conv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
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


class ConvB(object):

  @staticmethod
  def forward(x, w, b, conv_param):
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
         
    dx = dx[:,:,1:-1,1:-1] #delete padded "pixels"
    print(dx.shape)   

    return dx, dw, db

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

class DeepConvNetTorch(object):
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
               dtype=torch.float, device='cpu',
               _debug=False):
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
    self._debug = _debug
    self.debug_values= dict()
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
    self.slowpool = slowpool
    self.num_filters = num_filters
  
    if device == 'cuda':
      device = 'cuda:0'
    
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


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None

    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    slowpool_param = {'pool_height':2, 'pool_width':2, 'stride': 1}
    cache = {}
    out = X
    for i in range(self.num_layers-1):
      # print(i)
      # print(out.shape)
      if i in self.max_pools:
        # print('max_pool')
        if self.batchnorm:
          #print('batch_pool')
          if self._debug: print(f"\nImplement Layer: conv_bn_relu_pool{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape}     \t with {conv_param}")
          out,cache['{}'.format(i)] = Conv_BatchNorm_ReLU_Pool.forward(out, 
                                                                        self.params['W{}'.format(i)], 
                                                                        self.params['gamma{}'.format(i)], 
                                                                        self.params['beta{}'.format(i)], 
                                                                        conv_param, 
                                                                        self.bn_params[i],
                                                                        pool_param) 
          if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
          if self._debug: self.debug_values[f'conv_bn_relu_pool{i}']=out[0][0][0][0].detach().cpu().numpy()
          with open('./output_torch/conv_bn_relu_pool{}'.format(i), mode='w+') as f:
            f.write(str(out))
        else:  
          if self._debug: print(f"\nImplement Layer: conv_bn_pool{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape} \t with {conv_param}")
          out,cache['{}'.format(i)] = Conv_ReLU_Pool.forward(out,
                                                             self.params['W{}'.format(i)], 
                                                             conv_param,pool_param)
          if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
          if self._debug: self.debug_values[f'conv_bn_pool{i}']=out[0][0][0][0].detach().cpu().numpy()
          with open('./output_torch/conv_relu_pool{}'.format(i), mode='w+') as f:
            f.write(str(out))
      else:
        if self.slowpool and i == 6 :
          # print(self.num_filters[i])
          if self._debug: print(f"\nImplement Layer: pad_slowpool{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape} \t with {slowpool_param}")
          out = F.pad(out, (0, 1, 0, 1))
          out, _ = FastMaxPool.forward(out, slowpool_param)
          if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
          if self._debug: self.debug_values[f'pad_slowpool{i}']=out[0][0][0][0].detach().cpu().numpy()
          with open('./output_torch/pad_slowpool{}'.format(i), mode='w+') as f:
            f.write(str(out))
        if self.batchnorm:
          #print('batch_without_pool')
          if self._debug: print(f"\nImplement Layer: conv_bn_relu{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape} \t with {conv_param}")
          out,cache['{}'.format(i)] = Conv_BatchNorm_ReLU.forward(out,
                                                                  self.params['W{}'.format(i)], 
                                                                  self.params['gamma{}'.format(i)],
                                                                  self.params['beta{}'.format(i)],
                                                                  conv_param,self.bn_params[i])
          if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
          if self._debug: self.debug_values[f'conv_bn_relu{i}']=out[0][0][0][0].detach().cpu().numpy()
          with open('./output_torch/conv_bn_relu{}'.format(i), mode='w+') as f:
            f.write(str(out))
        else:
          if self._debug: print(f"\nImplement Layer: conv_relu{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape} \t with {conv_param}")
          out,cache['{}'.format(i)] = Conv_ReLU.forward(out, 
                                                        self.params['W{}'.format(i)], 
                                                        conv_param)
          if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
          if self._debug: self.debug_values[f'conv_relu{i}']=out[0][0][0][0].detach().cpu().numpy()
          with open('./output_torch/conv_relu{}'.format(i), mode='w+') as f:
            f.write(str(out))
    i+=1
    conv_param['pad'] = 0
    if self._debug: print(f"\nImplement Layer: conv_final{i}\nInput shape: {out.shape}\nLayer shape: {self.params['W{}'.format(i)].shape} \t with {conv_param}")
    out, cache['{}'.format(i)] = FastConvWB.forward(out, 
                                                    self.params['W{}'.format(i)], 
                                                    self.params['b{}'.format(i)], 
                                                    conv_param)
    if self._debug: print(f"Output feature sample value: {out[0][0][0][0].detach().cpu().numpy()}")
    if self._debug: self.debug_values[f'conv_final{i}']=out[0][0][0][0].detach().cpu().numpy()
    with open('./output_torch/conv_final{}'.format(i), mode='w+') as f:
        f.write(str(out))

    scores = out
    bsize = out.shape[0]
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

    xy_pred = torch.sigmoid(out[:, :, 0:2])
    conf_pred = torch.sigmoid(out[:, :, 4:5])
    hw_pred = torch.exp(out[:, :, 2:4])
    class_score = out[:, :, 5:]
    class_pred = F.softmax(class_score, dim=-1)
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)


    if y is None:
      return delta_pred, conf_pred, class_pred

    loss, grads = 0, {}

    loss, dout = softmax_loss(scores, y)
    for i in range(self.num_layers):
      if self.batchnorm:
        if i <= (self.num_layers-2):
          loss += (self.params['gamma{}'.format(i)]*self.params['gamma{}'.format(i)]).sum()*self.reg
      loss += (self.params['W{}'.format(i)]*self.params['W{}'.format(i)]).sum()*self.reg
    last_dout, dw, db  = FastConv.backward(dout, cache['{}'.format(i)])
    grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
    grads['b{}'.format(i)] = db
    for i in range(i-1,-1,-1):
      if i in self.max_pools:
        if self.batchnorm:
          last_dout, dw, db, dgamma1,dbetta1  = Conv_BatchNorm_ReLU_Pool.backward(last_dout, cache['{}'.format(i)])
          grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
          grads['b{}'.format(i)] = db
          grads['beta{}'.format(i)] = dbetta1
          grads['gamma{}'.format(i)] = dgamma1 + 2*self.params['gamma{}'.format(i)]*self.reg
        else:  
          last_dout, dw, db  = Conv_ReLU_Pool.backward(last_dout, cache['{}'.format(i)])
          grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
          grads['b{}'.format(i)] = db
      else:
        if self.batchnorm:
          last_dout, dw, db,dgamma1,dbetta1  = Conv_BatchNorm_ReLU.backward(last_dout, cache['{}'.format(i)])
          grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
          grads['b{}'.format(i)] = db
          grads['beta{}'.format(i)] = dbetta1
          grads['gamma{}'.format(i)] = dgamma1 + 2*self.params['gamma{}'.format(i)]*self.reg
        else:
          last_dout, dw, db  = Conv_ReLU.backward(last_dout, cache['{}'.format(i)])
          grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
          grads['b{}'.format(i)] = db


    return scores, loss, grads
  
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
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      # Replace "pass" statement with your code
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
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      # Replace "pass" statement with your code
      normolized = ((x - running_mean)/(running_var+ eps)**(1/2))
      out = normolized * gamma + beta
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code
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
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Replace "pass" statement with your code
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    N,D = dout.shape
  #get the dimensions of the input/output
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(xhat * dout, dim=0)
    dx = (gamma*ivar/N) * (N*dout - xhat*dgamma - dbeta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    N,C,H,W = x.shape
    pre_m = x.permute(1,0,2,3).reshape(C,-1).T
    pre_m_normolized, cache= BatchNorm.forward(pre_m, gamma, beta, bn_param)
    out = pre_m_normolized.T.reshape(C, N, H, W).permute(1,0,2,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    N,C,H,W = dout.shape
    pre_m = dout.permute(1,0,2,3).reshape(C,-1).T
    dx, dgamma, dbeta = BatchNorm.backward_alt(pre_m, cache)
    dx =dx.T.reshape(C, N, H, W).permute(1,0,2,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


################################################################################
################################################################################
#################   Fast Implementations and Sandwich Layers  ##################
################################################################################
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad, bias=False)
    layer.weight = torch.nn.Parameter(w)
    # layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      # db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw = torch.zeros_like(tx), torch.zeros_like(layer.weight)
    return dx, dw

class FastConvWB(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db

class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
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
    a, conv_cache = FastConv.forward(x, w, conv_param)
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
    dx, dw = FastConv.backward(da, conv_cache)
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
    a, conv_cache = FastConv.forward(x, w, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw = FastConv.backward(da, conv_cache)
    return dx, dw

class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw = FastConv.backward(da, conv_cache)
    return dx, dw, dgamma, dbeta

class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw = FastConv.backward(da, conv_cache)
    return dx, dw, dgamma, dbeta

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