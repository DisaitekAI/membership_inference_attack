import torch
import torch.nn as nn
import torch.nn.functional as F

class MIA_model(nn.Module):
  """Default MIA attack model
  """
  def __init__(self, input_size = None, hidden_size = None):
    """
    
    Args:
      input_size (int): size of the input vector.
      
      hidden_size (int): size of the hidden layer by default it is 2*input_size
      
    """
    super(MIA_model, self).__init__()
    
    if input_size is None:
      raise ValueError('MIA_model: input size is not set')
      
    if hidden_size is None:
      hidden_size = 2 * input_size
      
    self.dense1      = nn.Linear(input_size, hidden_size)           
    self.batch_norm  = nn.BatchNorm1d(hidden_size)
    self.dense2      = nn.Linear(hidden_size, 2)

  def forward(self, x):
    x = self.dense1(x)
    x = self.batch_norm(x)
    x = torch.tanh(x)
    x = self.dense2(x)
    return F.log_softmax(x, dim=1)
