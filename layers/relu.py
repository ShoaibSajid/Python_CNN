import numpy as np

class Relu:
  # A standard ReLu activation layer.

  def __init__(self):
    pass
    

  def forward(self, input):
    '''
    The forward propagation is used to calculate the output of a neural network given an input. 
    In the case of ReLU activation function, the output is calculated as the maximum of 0 and the input value
    '''
    
    # The input is passed through the np.maximum function which returns the element-wise maximum of 0 and the input values. 
    out = np.maximum(0, input)
    
    # The self.last_input is stored for use during backpropagation
    self.last_input = input
    
    # The output is returned.
    return out
  
  

  def backprop(self, d_L_d_out):
    '''
    The backward propagation is used to calculate the gradients of the 
    loss function with respect to the parameters of the neural network. 
    
    In the case of ReLU activation function, the gradient of the loss 
    with respect to the input is 1 for positive input values and 0 for negative input values
  
    The input dout is the gradient of the loss with respect to the 
    output of the ReLU layer. 
    '''
    
    # The self.last_input value x is retrieved from the 
    # forward propagation step. 
    x = self.last_input
    
    # The gradient with respect to the input is calculated as a copy of dout 
    # and set to 0 where the input value is negative.
    dx = d_L_d_out.copy()
    dx[x <= 0] = 0
    
    return dx
