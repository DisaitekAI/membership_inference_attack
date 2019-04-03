import torch
import torch.nn as nn
import torch.nn.functional as F

class MIA_model(nn.Module):
  def __init__(self, *opt_dict, **opt_args):
    """
    :input_size size of the input vector
    :hidden_size size of the hidden layer by default it is 5*input_size
    """
    super(MIA_model, self).__init__()
    
    self.input_size = 0
    self.hidden_size = 0
    
    for dictionary in opt_dict:
      for key in dictionary:
        setattr(self, key, dictionary[key])
    for key in opt_args:
      setattr(self, key, opt_args[key])
      
    if self.input_size == 0:
      print("MIA_model error: input size is not set")
      return
    else:
      if self.hidden_size == 0:
        self.hidden_size = 5 * self.input_size
      
    self.dense1 = nn.Linear(self.input_size, self.hidden_size)
    self.dense1_batch_norm = nn.BatchNorm1d(self.hidden_size)
    self.dense2 = nn.Linear(self.hidden_size, 2)

  def forward(self, x):
    x = torch.tanh(self.dense1_batch_norm(self.dense1(x)))
    x = self.dense2(x)
    return F.log_softmax(x, dim=1)
