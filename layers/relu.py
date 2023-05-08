import numpy as np

class Relu:
  # A standard fully-connected layer with softmax activation.

  def __init__(self):
    pass
    

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_totals = input
    
    out = input
    
    shape = input.shape
    if len(shape)==2:
      for i in range(shape[0]):
        for j in range(shape[1]):
          out[i,j]=max(0.0,input[i,j])

    if len(shape)==3:
      for i in range(shape[0]):
        for j in range(shape[1]):
          for k in range(shape[2]):
            out[i,j,k]=max(0.0,input[i,j,k])
    
    self.last_output = out
    return out
  
  

  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)
      if S==0: S=1

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
      
      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t
      
      return d_L_d_t
