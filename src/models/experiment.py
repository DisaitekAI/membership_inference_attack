import pathlib, sys
import pdb

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
data_src_path = home_path/'src'/'data'
utils_path = home_path/'src'/'utils'
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
# add ../data/ into the path
if data_src_path.as_posix() not in sys.path:
  sys.path.insert(0, data_src_path.as_posix())

from dataset_generator import Dataset_generator
from mnist_model import Mnist_model
from target import Target
from shadow_swarm_trainer import get_mia_dataset
from mia_model import MIA_model
from utils_modules import train, test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def experiment(academic_dataset         = None, 
               custom_target_model      = None,
               custom_target_optim_args = None,
               custom_mia_model         = None,
               custom_mia_optim_args    = None,
               use_cuda                 = False,
               mia_model_path           = None,
               target_model_path        = None,
               shadow_number            = 100,
               custom_shadow_model      = None,
               custom_shadow_optim_args = None,
               shadow_model_base_path   = None,
               mia_dataset_path         = None):
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
  
  :shadow_number 100 by default
  
  :shadow_model_base_path base file path for saving the shadow models. 
    File names will be incremented with the shadow index. Required.
  
  :custom_shadow_model an OrderedDict description of the 
    shadow model. None by default. If the target is online, providing 
    the model is required. If the target is offline by default the 
    shadow model is a copy of the target model.
  
  :custom_shadow_optim_args dict of custom values for lr and 
    momentum. None by default.
    
  :mia_dataset_path path for saving or loading the mia dataset. 
    Required.
  """
  if (mia_model_path is None) or (mia_dataset_path is None): 
    raise ValueError('experiment(): mia_model_path or mia_dataset_path is not set')
  
  device = torch.device('cuda' if use_cuda else 'cpu')
  cuda_args = { 'num_workers' : 1, 'pin_memory' : True } if use_cuda else {}
  
  shadow_swarm_dataset = None
  
  # train / load target model if offline
  if academic_dataset is not None:
    if target_model_path is None:
      raise ValueError('experiment(): target_model_path is not set')
      
    # the model has not been trained so we do it here
    if not pathlib.Path(target_model_path).exists():
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = True)
      train_set = dg.generate()
      train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True, **cuda_args)
      
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = False)
      test_set = dg.generate()
      shadow_swarm_dataset = test_set
      test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = True, **cuda_args)
      
      model = None
      if custom_target_model is None:
        if academic_dataset == 'mnist':
          model = Mnist_model().to(device)
      else:
        model = nn.Sequential(custom_target_model).to(device)
      
      optim_args = { 'lr' : 0.01, 'momentum' : 0.5 }
      if custom_target_optim_args is not None:
        optim_args = custom_target_optim_args
        
      optimizer = optim.SGD(model.parameters(), **optim_args)
      
      print('training the target model')
      for epoch in range(1, 5):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

      torch.save(model, target_model_path)
      
    if shadow_swarm_dataset is None:
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = False)
      shadow_swarm_dataset = dg.generate()
  
  shadow_model = None
  if custom_shadow_model is None:
    shadow_model = torch.load(target_model_path).to(device)
  else:
    shadow_model = nn.Sequential(custom_shadow_model).to(device)
    
  mia_dataset = get_mia_dataset(shadow_swarm_dataset, shadow_number, 
                                shadow_model, use_cuda, 
                                custom_shadow_optim_args,
                                shadow_model_base_path,
                                mia_dataset_path)
  
  train_size = int(0.8 * len(mia_dataset))
  test_size  = len(mia_dataset) - train_size
  train_set, test_set = torch.utils.data.random_split(mia_dataset, [train_size, test_size])

  mia_model = None
  if custom_mia_model is None:
    first_input_activations, _, _ = train_set[0]
    mia_model = MIA_model(input_size = len(first_input_activations))
  else:
    mia_model = nn.Sequential(custom_mia_model).to(device)
  
  train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True, **cuda_args)
  test_loader  = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = True, **cuda_args)
  
  optim_args = { 'lr' : 0.01, 'momentum' : 0.5 }
  if custom_mia_optim_args is not None:
    optim_args = custom_mia_optim_args
  optimizer = optim.SGD(mia_model.parameters(), **optim_args)

  print('training the MIA model')
  for epoch in range(1, 5):
    train(mia_model, device, train_loader, optimizer, epoch)
    test(mia_model, device, test_loader)

  torch.save(mia_model, mia_model_path)

