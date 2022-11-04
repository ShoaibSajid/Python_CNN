import numpy as np

class FC:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)
    np.save('linear_filters',self.weights)
    np.save('linear_biases' ,self.biases)
    

  def forward(self, input):
    '''
    Performs a forward pass of the FC layer using the given input.
    Returns a 1d numpy array containing the respective values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape
    input = input.flatten()
    self.last_input = input

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals

    return totals
  
  

  def backprop(self, d_L_d_t, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    # for i, gradient in enumerate(d_L_d_out):
    #   if gradient == 0:
    #     continue

    # Gradients of totals against weights/biases/input
    d_t_d_w = self.last_input
    d_t_d_b = 1
    d_t_d_inputs = self.weights

    # Gradients of loss against weights/biases/input
    d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
    d_L_d_b = d_L_d_t * d_t_d_b
    d_L_d_inputs = d_t_d_inputs @ d_L_d_t

    # Update weights / biases
    self.weights -= learn_rate * d_L_d_w
    self.biases -= learn_rate * d_L_d_b

    return d_L_d_inputs.reshape(self.last_input_shape)
