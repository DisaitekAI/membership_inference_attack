import pathlib, sys

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
from shadow_swarm_trainer import get_mia_train_dataset, get_mia_test_dataset
from mia_model import MIA_model
from utils_modules import train, test
from miscellaneous import fixed_random_split, BalancedSampler
from statistics import Statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

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
               mia_train_dataset_path   = None,
               mia_test_dataset_path    = None,
               class_number             = None,
               stats                    = None):
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
    
  :mia_train_dataset_path base path for saving or loading the mia dataset 
    used for training. Required.
    
  :mia_test_dataset_path base path for saving or loading the mia dataset 
    used for testing. Required.
    
  :class_number number of classes in the target model. Required.
    
  :stats pass a Statistics object to record stats for this experiment
  """
  
  if (mia_model_path         is None) or \
     (mia_train_dataset_path is None) or \
     (class_number           is None) or \
     (shadow_model_base_path is None) or \
     (mia_test_dataset_path  is None): 
    raise ValueError('experiment(): one of the required argument is not set')
  
  device = torch.device('cuda' if use_cuda else 'cpu')
  cuda_args = { 'num_workers' : 1, 'pin_memory' : True } if use_cuda else {}
  
  shadow_swarm_dataset = None
  shadow_dir = pathlib.Path(shadow_model_base_path).parent
  if not shadow_dir.exists():
    shadow_dir.mkdir()
  
  # train / load target model if offline
  if academic_dataset is not None:
    if target_model_path is None:
      raise ValueError('experiment(): target_model_path is not set')
      
    # the model has not been trained so we do it here
    if not pathlib.Path(target_model_path).exists():
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = True)
      train_set = dg.generate()
      # we take only the first half of the dataset to train the model
      # the second half is used to train the shadow models
      half_len = int(len(train_set) / 2)
      train_set, shadow_swarm_dataset = fixed_random_split(train_set, [half_len, len(train_set) - half_len])
      
      train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True, **cuda_args)
      
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = False)
      test_set = dg.generate()
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
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = True)
      train_set = dg.generate()
      half_len = int(len(train_set) / 2)
      _, shadow_swarm_dataset = fixed_random_split(train_set, [half_len, len(train_set) - half_len])
  
  shadow_model = None
  if custom_shadow_model is None:
    shadow_model = torch.load(target_model_path).to(device)
  else:
    shadow_model = nn.Sequential(custom_shadow_model).to(device)
    
  mia_train_datasets = get_mia_train_dataset(shadow_swarm_dataset, shadow_number, 
                                             shadow_model, use_cuda, 
                                             custom_shadow_optim_args,
                                             shadow_model_base_path,
                                             mia_train_dataset_path,
                                             class_number)
  
  # TODO(PI) only for mnist right now
  dg = Dataset_generator(method = 'academic', name = academic_dataset, train = True)
  train_set = dg.generate()
  half_len = int(len(train_set) / 2)
  train_set, _ = fixed_random_split(train_set, [half_len, len(train_set) - half_len])
  
  dg = Dataset_generator(method = 'academic', name = academic_dataset, train = False)
  test_set = dg.generate()
  
  target_model = None
  if custom_target_model is None:
    target_model = torch.load(target_model_path).to(device)
  else:
    target_model = nn.Sequential(custom_target_model).to(device)
    
  mia_test_datasets = get_mia_test_dataset(train_set,
                                           test_set,
                                           target_model,
                                           use_cuda,
                                           mia_test_dataset_path,
                                           class_number)  
  # ~ train_size = int(0.8 * len(mia_dataset))
  # ~ test_size  = len(mia_dataset) - train_size
  # ~ train_set, test_set = torch.utils.data.random_split(mia_dataset, [train_size, test_size])
  mia_models = list()
  if custom_mia_model is None:
    for i in range(class_number):
      mia_models.append(MIA_model(input_size = class_number))
  else:
    for i in range(class_number):
      mia_models.append(nn.Sequential(custom_mia_model).to(device))
  
  # correction for class imbalance
  # ~ weights = torch.tensor([2/shadow_number, 2 * (shadow_number-1)/shadow_number])
  weights = None
  
  # ~ train_sampler = WeightedRandomSampler(weights, num_samples = len(train_set), 
                                        # ~ replacement = True)
  # ~ train_loader = torch.utils.data.DataLoader(train_set, batch_size = 2 * shadow_number, 
                                             # ~ shuffle = False, **cuda_args, 
                                             # ~ sampler = train_sampler)
                                             
  # ~ train_sampler = WeightedRandomSampler(weights, num_samples = len(test_set), 
                                        # ~ replacement = True)                                          
  # ~ test_loader  = torch.utils.data.DataLoader(test_set, batch_size = 1000, 
                                             # ~ shuffle = False, **cuda_args,
                                             # ~ sampler = train_sampler)
                                             
  optim_args = { 'lr' : 0.01, 'momentum' : 0.5 }
  if custom_mia_optim_args is not None:
    optim_args = custom_mia_optim_args
  
  mia_model_dir = pathlib.Path(mia_model_path)
  if not mia_model_dir.exists():
    mia_model_dir.mkdir()
  
  for i in range(class_number):
    print(f"training the MIA model for class {i}")  
    optimizer = optim.SGD(mia_models[i].parameters(), **optim_args)
    
    balanced_train_dataset = BalancedSampler(mia_train_datasets[i])
    train_loader = torch.utils.data.DataLoader(mia_train_datasets[i], batch_size = 32, 
                                               sampler = balanced_train_dataset, **cuda_args)    
                                                        
    balanced_test_dataset = BalancedSampler(mia_test_datasets[i])
    test_loader  = torch.utils.data.DataLoader(mia_test_datasets[i], batch_size = 1000, 
                                               sampler = balanced_test_dataset, **cuda_args)
                                               
    # ~ train_loader = torch.utils.data.DataLoader(mia_train_datasets[i], batch_size = 32, 
                                               # ~ shuffle = False, **cuda_args)  
    # ~ test_loader  = torch.utils.data.DataLoader(mia_test_datasets[i], batch_size = 1000, 
                                               # ~ shuffle = False, **cuda_args)                                           
    for epoch in range(1, 5):
      train(mia_models[i], device, train_loader, optimizer, epoch, 
            class_weights = weights)
            
      test(mia_models[i], device, test_loader, 
           class_weights = weights, test_stats = stats)
           
      stats.print_results()
    
    torch.save(mia_models[i], (mia_model_dir/f"class_{i}.pt").as_posix())
