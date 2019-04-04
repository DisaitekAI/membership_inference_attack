import torch
import torch.nn as nn
import torch.nn.functional as F

class MIA_model(nn.Module):
  def __init__(self, input_size = None, hidden_size = None):
    """
    :input_size size of the input vector
    :hidden_size size of the hidden layer by default it is 2*input_size
    """
    super(MIA_model, self).__init__()
    
    if input_size is None:
      raise ValueError('MIA_model: input size is not set')
      
    if hidden_size is None:
      hidden_size = 2 * input_size
      
    self.dense1_activations      = nn.Linear(input_size, hidden_size)           
    self.dense1_targetclass      = nn.Linear(input_size, hidden_size)
    self.batch_norm_activations  = nn.BatchNorm1d(hidden_size)
    self.batch_norm_targetclass  = nn.BatchNorm1d(hidden_size)
    self.dense2                  = nn.Linear(2 * hidden_size, 2)

  def forward(self, activations, targetclass):
    activations = self.dense1_activations(activations)
    activations = self.batch_norm_activations(activations)
    activations = torch.tanh(activations)
    targetclass = self.dense1_targetclass(targetclass)
    targetclass = self.batch_norm_targetclass(targetclass)
    targetclass = torch.tanh(activations)
    x           = torch.cat((activations, targetclass), 1)
    x           = self.dense2(x)
    return F.log_softmax(x, dim=1)
