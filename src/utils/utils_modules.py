import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Flatten(nn.Module):
  """flatten a dimension of a tensor
  """
  def forward(self, x):
    return x.view(x.size()[0], -1)
    
from statistics import Statistics

class Print(nn.Module):
  """print the inputs and/or the shape of a layer. Used to debug models 
  built with an OrderedDict.  
  """
  def __init__(self, title = None, print_inputs = False, print_shape = True):
    """
    Args:
      title (string): title to print before the data.
      
      print_inputs (bool): whether to print the input values of the layer or not. False by default.
      
      print_shape (bool): whether to print the shape of the input or not. True by default.
    """
    super(Print, self).__init__()
    
    self.title        = title
    self.print_inputs = print_inputs
    self.print_shape  = print_shape
    
  def forward(self, x):
    if self.title is not None:
      print("\nPrint layer: " + self.title + "\n")
    if self.print_inputs:
      print(x)
    if self.print_shape:
      print(x.shape)
    return x

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

def train(model, device, train_loader, optimizer, epoch, verbose = True, class_weights = None, train_stats = None):
  """train a model
  
  Args:
    model (torch Module): the model to be trained.
    
    device (torch Device): cpu or gpu use.
    
    train_loader (torch DataLoader): iterator of the train batches.
    
    optimizer (torch Optimizer): learning algorithm.
    
    epoch (int): epoch of the training algorithm.
    
    verbose (bool): whether this function display results or not. True by default.
    
    class_weights (list(float)): weights to apply to each class to process the loss. 
  """
  model.train()
  for batch_idx, batch in enumerate(train_loader):
    input_list = batch[0:-1]
    target = batch[-1]
    input_list = [e.to(device) for e in input_list]
    target = target.to(device)
    
    # protect batchnormalized models against batch size of 1
    if len(target) == 1:
      continue
      
    optimizer.zero_grad()
    output = model(*input_list)
    
    loss = F.nll_loss(output, target, weight = class_weights)
    # ~ loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    
    if verbose and batch_idx % 10 == 0:
      size = 0
      for data in input_list:
        size += len(data)
    
      print('  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * size, len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

    if train_stats is not None:
      train_stats.add_loss(loss.item())

    # if batch_idx == 2:
    #   break
        
      
        
def test(model, device, test_loader, verbose = True, class_weights = None, test_stats = None, name = "test set"):
  """test a model
  
  Args:
    model (torch Module): the model to be trained.
    
    device (torch Device): cpu or gpu use.
    
    test_loader (torch DataLoader): iterator of the test batches.
    
    verbose (bool): whether this function display results or not. True by default.
    
    class_weights (list(float)): weights to apply to each class to process the loss. 
    
    test_stats (Statistics): the statistics object that record results.
  """
  model.eval()
  test_loss = 0
  correct = 0
  
  with torch.no_grad():
    for batch_idx,batch in enumerate(test_loader):
      input_list = batch[0:-1]
      target = batch[-1]
      input_list = [e.to(device) for e in input_list]
      target = target.to(device)

      output = model(*input_list)
      test_loss += F.nll_loss(output, target, reduction = 'sum', weight = class_weights).item() # sum up batch loss
      # ~ test_loss += torch.nn.CrossEntropyLoss()(output, target)
      pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
      
      correct += pred.eq(target.view_as(pred)).sum().item()

      if test_stats is not None:
        test_stats.new_batch(pred.view_as(target).tolist(), target.tolist())

      # if batch_idx == 2:
      #   break
            
  test_loss /= len(test_loader.dataset)

  if verbose:
    print('\n  {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          name, test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
