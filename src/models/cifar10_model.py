import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10_model(nn.Module):
  """
  Bad CNN for the Cifar10 dataset
  """
  def __init__(self):
    super(Cifar10_model, self).__init__()
    self.conv1              = nn.Conv2d(3, 32, 3, 1)
    self.conv1_dropout      = nn.Dropout(0.25)
    self.conv2              = nn.Conv2d(32, 64, 3, 1)
    self.conv2_dropout      = nn.Dropout(0.25)
    self.dense1             = nn.Linear(6*6*64, 512)
    self.dense1_dropout     = nn.Dropout(0.5)
    self.dense2             = nn.Linear(512, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = self.conv1_dropout(x)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = self.conv2_dropout(x)
    x = x.view(-1, 6*6*64)
    x = F.relu(self.dense1(x))
    x = self.dense1_dropout(x)
    x = self.dense2(x)
    return F.log_softmax(x, dim=1)
