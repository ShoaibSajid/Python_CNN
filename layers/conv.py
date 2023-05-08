import numpy as np
from pathlib import Path
Path('weights/temp/conv').mkdir( parents=True, exist_ok=True )
'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    np.save('weights/temp/conv//conv_filters',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
      
    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None


class Conv3x3_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    np.save('weights/temp/conv//conv_filters',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:]
    h, w = image_padded.shape[:2]

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image_padded[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def iterate_regions_3d(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:,1:-1]
    h, w, c = image_padded.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image_padded[i:(i + 3), j:(j + 3), :]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w, c = input.shape
    output = np.zeros((h , w , self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
      
    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''

    # d_L_d_weights = d_L_d_out * inputs
    # -------------------------------------------------------------
    d_L_d_filters = np.zeros(self.filters.shape)
    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region



    # d_L_d_input   = d_L_d_out * weights (convolve)
    # -------------------------------------------------------------
    d_L_d_input = np.zeros(self.last_input.shape)
    for im_region, i, j in self.iterate_regions_3d(d_L_d_out):
      for f in range(self.num_filters):
        d_L_d_input[i,j] += np.sum(  self.filters[f,:,:] * im_region[:,:,f]  )
    
    
    
    
    # Update filters
    self.filters -= learn_rate * d_L_d_filters
    
    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return d_L_d_input





class Conv3x3_1_to_n_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, output=1, input=1):
    num_filters = output
    in_ch = input
    
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    np.save('weights/temp/conv//conv_filters_3d',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)
    h, w = image_padded.shape[:2]
    for i in range(h - 2):
      for j in range(w - 2):
          im_region = image_padded[i:(i + 3), j:(j + 3)]
          yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h, w , self.num_filters))
    # output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)
    d_L_d_input   = np.zeros(self.last_input.shape) 

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
    
    for out_ch in range(self.num_filters):
      d_L_d_input[:,:] += d_L_d_out[:,:, out_ch] * self.last_input[:,:]
        
        
    # errors of previous layer = weights_of_this_layer-T * errors of this layer
    
    # Update filters
    self.filters -= learn_rate * d_L_d_filters
    
    return d_L_d_input










class Conv3x3_n_to_n_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, output=1, input=1):
    num_filters = output
    in_ch = input
    
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3, in_ch) / 9
    np.save('weights/temp/conv//conv_filters_3d',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:,1:-1]
    h, w = image_padded.shape[:2]
    for i in range(h - 2):
      for j in range(w - 2):
          im_region = image_padded[i:(i + 3), j:(j + 3), :]
          yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w, c = input.shape
    output = np.zeros((h, w , self.num_filters))
    # output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      for filter in range(self.num_filters):
        output[i, j, filter] = np.sum(im_region * self.filters[filter,:,:,:], axis=(0, 1, 2))

    
    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
    
    
    
    d_L_d_input   = np.zeros(self.last_input.shape) 
      
    # Method 4
    for im_region, i, j in self.iterate_regions(d_L_d_out):
      for in_ch in range(d_L_d_input.shape[-1]):
        # d_L_d_input[i,j,in_ch] += np.sum ( im_region[:,:,:] * np.transpose( self.filters[:,:,:,in_ch]) , axis=(0,1,2) )
        d_L_d_input[i,j,in_ch] += np.sum( np.matmul( im_region[:,:,:] , np.transpose( self.filters[:,:,:,in_ch] , axes=(2,0,1)) ) , axis=(0,1,2) )
  

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    return d_L_d_input
