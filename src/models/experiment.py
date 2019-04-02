import os, sys, inspect

project_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) + "/../"
data_dir = project_dir + "/data/"

# add ../data/ into the path
if data_dir not in sys.path:
  sys.path.insert(0, data_dir)

from dataset_generator import Dataset_generator
from mnist_model import Mnist_model
from target import Target
# ~ from shadow_swarm_trainer import Shadow_swarm_trainer
from mia_model import MIA_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
        
def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def experiment(*opt_dict, **opt_args):
  """
  
  start a membership inference attack experiment
  
  :academic_dataset name of the academic dataset used. None by default.
  
  :custom_target_model an OrderedDict description of the 
    target model. None by default.
    
  :custom_target_optim_args dict of custom values for lr and 
    momentum. None by default.
    
  :custom_mia_model an OrderedDict description of the mia model. 
    None by default.
    
  :custom_mia_optim_args dict of custom values for lr and 
    momentum. None by default.
    
  :target_model_path  required if the target is offline
   
  :mia_model_path required
  
  :use_cuda False by default
    
  """
  all_opts = dict()
  all_opts["academic_dataset"] = None
  all_opts["custom_target_model"] = None
  all_opts["custom_target_optim_args"] = None
  all_opts["custom_mia_model"] = None
  all_opts["custom_mia_optim_args"] = None
  all_opts["use_cuda"] = False
  
  for dictionary in opt_dict:
    for key in dictionary:
      all_opts[key] = dictionary[key]
  for key in opt_args:
    all_opts[key] = opt_args[key]
  
  if "mia_model_path" not in all_opts: 
    print("experiment() error: mia_model_path is not set")
    return
  
  device = torch.device("cuda" if all_opts["use_cuda"] else "cpu")
  cuda_args = { 'num_workers' : 1, 'pin_memory' : True} if all_opts["use_cuda"] else {}
  
  target = None
  shadow_swarm_dataset = None
  
  # train / load target model if offline
  if all_opts["academic_dataset"] != None:
    if "target_model_path" not in all_opts:
      print("experiment() error: target_model_path is not set")
      return
      
    # the model has not been trained so we do it here
    if not os.path.isfile(all_opts["target_model_path"]):
      dg = Dataset_generator(method = "academic", name = all_opts["academic_dataset"], train = True)
      train_set = dg.generate()
      train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True, **cuda_args)
      
      dg = Dataset_generator(method = "academic", name = all_opts["academic_dataset"], train = False)
      test_set = dg.generate()
      shadow_swarm_dataset = test_set
      test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = True, **cuda_args)
      
      model = None
      if all_opts["custom_target_model"] != None:
        model = nn.Sequential(all_opts["custom_target_model"]).to(device)
      else:
        if all_opts["academic_dataset"] == "mnist":
          model = Mnist_model().to(device)
      
      optim_args = all_opts["custom_target_optim_args"] if all_opts["custom_target_optim_args"] else { 'lr' : 0.01, 'momentum' : 0.5 }
      optimizer = optim.SGD(model.parameters(), **optim_args)
      
      print("training the target model")
      for epoch in range(1, 5):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

      torch.save(model, all_opts["target_model_path"])
      
    target = Target(model_path = all_opts["target_model_path"])
    if shadow_swarm_dataset is None:
      dg = Dataset_generator(method = "academic", name = all_opts["academic_dataset"], train = False)
      shadow_swarm_dataset = dg.generate()
  
  # ~ # online attack case, not implemented yet
  # ~ if shadow_swarm_dataset is None
    # ~ dg = Dataset_generator(method = "random")
    # ~ shadow_swarm_dataset = dg.generate()
    
  sst = Shadow_swarm_trainer(shadow_swarm_dataset)
  mia_dataset = sst.get_mia_dataset()
  
  train_size = int(0.8 * len(mia_dataset))
  test_size = len(mia_dataset) - train_size
  train_set, test_set = torch.utils.data.random_split(mia_dataset, [train_size, test_size])
  
  mia_model = None
  if all_opts["custom_mia_model"] != None:
    mia_model = nn.Sequential(all_opts["custom_mia_model"]).to(device)
  else:
    _, first_output = train_set[0]
    mia_model = MIA_model(input_size = len(first_output))
  
  train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True, **cuda_args)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = True, **cuda_args)
  
  optim_args = all_opts["custom_mia_optim_args"] if all_opts["custom_mia_optim_args"] else { 'lr' : 0.01, 'momentum' : 0.5 }
  optimizer = optim.SGD(model.parameters(), **optim_args)
  
  print("training the MIA model")
  for epoch in range(1, 5):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

  torch.save(model, all_opts["mia_model_path"])
  
  
    
  
