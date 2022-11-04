import numpy as np

class Softmax_test:
  # A standard fully-connected layer with softmax activation.

  def __init__(self):
    pass

  def forward(self, x):
      # self.input is a vector of length 10
      # and is the output of 
      # (w * x) + b
      # self.value = self.softmax(self.input)
      
      """Compute the softmax of vector x."""
      exps = np.exp(x)
      out = exps / np.sum(exps)
      
      self.input = x
      self.value = out
      
      return out

  def backward(self, x=[]):
    
      self.gradient = np.zeros((len(self.value),len(self.input)))
      
      if not x==[]:
        self.input = x
        self.value = self.forward(self.input)
    
      for i in range(len(self.value)):
          for j in range(len(self.input)):
              if i == j:
                self.gradient[i,j] = self.value[i] * (1-self.value[i])
              else: 
                self.gradient[i,j] = -self.value[i] * self.value[j]
                
      return self.gradient

                  