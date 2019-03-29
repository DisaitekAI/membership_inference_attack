import torch
import torch.nn as nn
import torch.nn.functional as F

class Mnist_model(nn.Module):
  def __init__(self):
    super(Mnist_model, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.conv2_batch_norm = nn.BatchNorm2d(50)
    self.dense1 = nn.Linear(4*4*50, 500)
    self.dense1_batch_norm = nn.BatchNorm1d(500)
    self.dense2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2_batch_norm(self.conv2(x)))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    x = torch.tanh(self.dense1_batch_norm(self.dense1(x)))
    x = self.dense2(x)
    return F.log_softmax(x, dim=1)
