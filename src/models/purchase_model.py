import torch
import torch.nn as nn
import torch.nn.functional as F

class Purchase_model(nn.Module):
  def __init__(self):
    super(Purchase_model, self).__init__()
    
    input_size  = 93
    hidden_size = input_size
    
    self.dropout     = nn.Dropout(0.5)
    self.batch_norm  = nn.BatchNorm1d(input_size)
    self.dense1      = nn.Linear(input_size, hidden_size) 
    self.dense2      = nn.Linear(hidden_size, hidden_size)
    self.dense3      = nn.Linear(hidden_size, 2)

  def forward(self, x):
    x = self.dropout(x)
    x = self.batch_norm(x)
    x = self.dense1(x)
    x = torch.tanh(x)
    x = self.dense2(x)
    x = torch.tanh(x)
    x = self.dense3(x)
    return F.log_softmax(x, dim=1)
