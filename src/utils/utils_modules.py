
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Flatten(nn.Module):
  """
  Flatten a dimension of a tensor
  """
  def forward(self, x):
    return x.view(x.size()[0], -1)

def weight_init(m):
  """
  Init the weights of all modules from a model
  
  Usage:
      model = Model()
      model.apply(weight_init)
  """
  if isinstance(m, nn.Conv1d):
    init.normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.Conv2d):
    init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.Conv3d):
    init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.ConvTranspose1d):
    init.normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.ConvTranspose2d):
    init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.ConvTranspose3d):
    init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        init.normal_(m.bias.data)
  elif isinstance(m, nn.BatchNorm1d):
    init.normal_(m.weight.data, mean=1, std=0.02)
    init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
    init.normal_(m.weight.data, mean=1, std=0.02)
    init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm3d):
    init.normal_(m.weight.data, mean=1, std=0.02)
    init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
    init.xavier_normal_(m.weight.data)
    init.normal_(m.bias.data)
  elif isinstance(m, nn.LSTM):
    for param in m.parameters():
      if len(param.shape) >= 2:
        init.orthogonal_(param.data)
      else:
        init.normal_(param.data)
  elif isinstance(m, nn.LSTMCell):
    for param in m.parameters():
      if len(param.shape) >= 2:
        init.orthogonal_(param.data)
      else:
        init.normal_(param.data)
  elif isinstance(m, nn.GRU):
    for param in m.parameters():
      if len(param.shape) >= 2:
        init.orthogonal_(param.data)
      else:
        init.normal_(param.data)
  elif isinstance(m, nn.GRUCell):
    for param in m.parameters():
      if len(param.shape) >= 2:
        init.orthogonal_(param.data)
      else:
        init.normal_(param.data)
        
def train(model, device, train_loader, optimizer, epoch, verbose = True):
  """
  train a model
  """
  model.train()
  for batch_idx, batch in enumerate(train_loader):
    input_list = batch[0:-1]
    target = batch[-1]
    input_list = [e.to(device) for e in input_list]
    target = target.to(device)
  
    optimizer.zero_grad()
    output = model(*input_list)

    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    
    if verbose and batch_idx % 10 == 0:
      size = 0
      for data in input_list:
        size += len(data)
    
      print('  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * size, len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
        
def test(model, device, test_loader, verbose = True):
  """
  test a model
  """
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for batch in test_loader:
      input_list = batch[0:-1]
      target = batch[-1]
      input_list = [e.to(device) for e in input_list]
      target = target.to(device)
      
      output = model(*input_list)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  
  if verbose:
    print('\n  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
