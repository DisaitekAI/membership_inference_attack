import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
data_src_path = home_path/'src'/'data'
utils_path = home_path/'src'/'utils'
viz_path = home_path/'src'/'visualization'
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
# add ../data/ into the path
if data_src_path.as_posix() not in sys.path:
  sys.path.insert(0, data_src_path.as_posix())
  
# add ../visualization/ into the path
if viz_path.as_posix() not in sys.path:
  sys.path.insert(0, viz_path.as_posix())

from dataset_generator import Dataset_generator
from mnist_model import Mnist_model
from purchase_model import Purchase_model
from cifar10_model import Cifar10_model
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

def cache_handling(no_cache                   = False,
                   no_mia_train_dataset_cache = False,
                   no_mia_test_dataset_cache  = False,
                   no_target_model_cache      = False,
                   no_mia_models_cache        = False,
                   no_shadow_cache            = False,
                   mia_model_path             = None,
                   target_model_path          = None,
                   shadow_model_base_path     = None,
                   mia_train_dataset_path     = None,
                   mia_test_dataset_path      = None):
  """cache_handling() control the cache usage of the MIA experiment
  
  The following elements are cached:
    * The trained target model (if the target model is offline)
    * The trained shadow models
    * The MIA training dataset built by the shadow models
    * The MIA test dataset built by the target model
    * The Trained MIA attack models
    
  Except for the trained MIA attack models, if the corresponding file 
  is detected, the default experiment behavior is to load its data. 
  Therefore this fonction control the removal of the cached data in 
  order to redo some parts of the experiment with different 
  hyperameters. All the parameters should be direct copy of the one 
  passed by experiment().
  
  Args:
    no_cache (bool): set to True to remove all cached data. False by default.
    
    no_mia_train_dataset_cache (bool): set to True to remove the cached MIA training dataset. False by default.
    
    no_mia_test_dataset_cache (bool): set to True to remove the cached MIA test dataset. False by default.
    
    no_target_model_cache (bool): set to True to remove the cached target model. False by default.
        
    no_mia_models_cache (bool): set to True to remove the cached target models. False by default.
          
    no_shadow_cache (bool): set to True to remove the cached shadow models. False by default.       
        
    mia_model_path (string/Path): path of the MIA models save dir.      
         
    target_model_path (string/Path): path of the target model save file.    
         
    shadow_model_base_path (string/Path): path of the shadow models save dir.     
    
    mia_train_dataset_path (string/Path): path of the MIA training datasets save dir. 
      
    mia_test_dataset_path (string/Path): path of the MIA test datasets save dir.    
    
  """
  import shutil
  
  if no_cache:
    no_mia_train_dataset_cache = True
    no_mia_test_dataset_cache = True
    no_target_model_cache = True
    no_mia_models_cache = True
    no_shadow_cache = True
    
  if no_mia_train_dataset_cache:
    path = pathlib.Path(mia_train_dataset_path)
    if path.exists():
      shutil.rmtree(path)
    
  if no_mia_test_dataset_cache:
    path = pathlib.Path(mia_test_dataset_path)
    if path.exists():
      shutil.rmtree(path)
    
  if no_target_model_cache:
    path = pathlib.Path(target_model_path)
    if path.exists():
      path.unlink()
    
  if no_mia_models_cache:
    path = pathlib.Path(mia_model_path)
    if path.exists():
      shutil.rmtree(path)
    
  if no_shadow_cache:
    path = pathlib.Path(shadow_model_base_path)
    
    base_dir = path.parent
    base_file_name = path.name
  
    models = list(base_dir.glob(base_file_name + '_*.pt'))
    for model in models:
      model.unlink()
    
def experiment(academic_dataset           = None, 
               custom_target_model        = None,
               custom_target_optim_args   = {},
               custom_mia_model           = None,
               custom_mia_optim_args      = None,
               use_cuda                   = False,
               mia_model_path             = None,
               target_model_path          = None,
               shadow_number              = 100,
               custom_shadow_model        = None,
               custom_shadow_optim_args   = {},
               shadow_model_base_path     = None,
               mia_train_dataset_path     = None,
               mia_test_dataset_path      = None,
               class_number               = None,
               stats                      = None,
               target_train_epochs        = 5,
               shadow_train_epochs        = 5,
               mia_train_epochs           = 5,
               no_cache                   = False,
               no_mia_train_dataset_cache = False,
               no_mia_test_dataset_cache  = False,
               no_target_model_cache      = False,
               no_mia_models_cache        = False,
               no_shadow_cache            = False,
               target_batch_size          = 64,
               shadow_batch_size          = None,
               mia_batch_size             = 32):
  """ experiment() starts a membership inference attack experiment
  
  Args:
    academic_dataset (string): name of the academic dataset to use. Currently datasets implemented are "cifar10", "purchase", "mnist".
    
    custom_target_model (OrderedDict): a custom target model to use instead of the one provided by default. Use an OrderedDict to describe the neural network architecture.
    
    custom_target_optim_args (Dict): arguments of the Adam optimizer to use instead of the default parameters.
    
    custom_mia_model (OrderedDict): a custom mia attack model to use instead of the one provided by default. Use an OrderedDict to describe the neural network architecture.
    
    custom_mia_optim_args (Dict): arguments of the Adam optimizer to use instead of the default parameters.
    
    use_cuda (bool): weither to use cuda or not. False by default. 
    
    mia_model_path (string/Path): path of the MIA models save dir.      
         
    target_model_path (string/Path): path of the target model save file.
    
    shadow_number (int): number of shadow models to use. 100 by default.
    
    custom_shadow_model (OrderedDict): a custom shadow model to use instead of the copy of the target model. Use an OrderedDict to describe the neural network architecture.
    
    custom_shadow_optim_args (Dict): arguments of the Adam optimizer to use instead of the default parameters.
    
    shadow_model_base_path (string/Path): path of the shadow models save dir.     
    
    mia_train_dataset_path (string/Path): path of the MIA training datasets save dir. 
      
    mia_test_dataset_path (string/Path): path of the MIA test datasets save dir.
    
    class_number (int): number class in the target problem. For instance, 10 for Mnist. 
    
    stats (Statistics): the statistics object used to record the experiment data.
    
    target_train_epochs (int): number epoch of the train function. 5 by default.
    
    shadow_train_epochs (int): number epoch of the train function. 5 by default.
    
    mia_train_epochs (int): number epoch of the train function. 5 by default.
    
    no_cache (bool): set to True to remove all cached data. False by default.
    
    no_mia_train_dataset_cache (bool): set to True to remove the cached MIA training dataset. False by default.
    
    no_mia_test_dataset_cache (bool): set to True to remove the cached MIA test dataset. False by default.
    
    no_target_model_cache (bool): set to True to remove the cached target model. False by default.
        
    no_mia_models_cache (bool): set to True to remove the cached target models. False by default.
          
    no_shadow_cache (bool): set to True to remove the cached shadow models. False by default. 
    
    target_batch_size (int): size of the target training batches. 64 by default.
    
    shadow_batch_size (int): size of the shadow training batches. Same value as target_batch_size by default.
    
    mia_batch_size (int): size of the mia attack model training batches. 32 by default. 
  
  """
  
  if (mia_model_path         is None) or \
     (mia_train_dataset_path is None) or \
     (class_number           is None) or \
     (shadow_model_base_path is None) or \
     (mia_test_dataset_path  is None): 
    raise ValueError('experiment(): one of the required argument is not set')
  
  cache_handling(no_cache, no_mia_train_dataset_cache, 
                 no_mia_test_dataset_cache, no_target_model_cache, 
                 no_mia_models_cache, no_shadow_cache, mia_model_path, 
                 target_model_path, shadow_model_base_path, 
                 mia_train_dataset_path, mia_test_dataset_path)
                 
  if shadow_batch_size is None:
    shadow_batch_size = target_batch_size
  
  device = torch.device('cuda' if use_cuda else 'cpu')
  cuda_args = { 'num_workers' : 1, 'pin_memory' : False } if use_cuda else {}
  
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
      
      train_loader = torch.utils.data.DataLoader(train_set, batch_size = target_batch_size, 
                                                 shuffle = True, **cuda_args)
      
      dg = Dataset_generator(method = 'academic', name = academic_dataset, train = False)
      test_set = dg.generate()
      test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = True, **cuda_args)
      
      model = None
      if custom_target_model is None:
        if academic_dataset == 'mnist':
          model = Mnist_model().to(device)
        if academic_dataset == 'purchase':
          model = Purchase_model().to(device)
        if academic_dataset == 'cifar10':
          model = Cifar10_model().to(device)
      else:
        model = nn.Sequential(custom_target_model).to(device)
      
      optim_args = {}
      if custom_target_optim_args is not None:
        optim_args = custom_target_optim_args
        
      optimizer = optim.Adam(model.parameters(), **optim_args)
      
      print('training the target model')
      stats.new_train(name = 'target model')
      for epoch in range(target_train_epochs):
        stats.new_epoch()
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, test_stats = stats)

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
                                             class_number, stats, 
                                             shadow_train_epochs,
                                             shadow_batch_size)
  
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

  stats.membership_distributions(mia_train_datasets, mia_test_datasets)
                                           
  mia_models = list()
  if custom_mia_model is None:
    for i in range(class_number):
      mia_models.append(MIA_model(input_size = class_number).to(device))
  else:
    for i in range(class_number):
      mia_models.append(nn.Sequential(custom_mia_model).to(device))
                                             
  optim_args = {}
  if custom_mia_optim_args is not None:
    optim_args = custom_mia_optim_args
  
  mia_model_dir = pathlib.Path(mia_model_path)
  if not mia_model_dir.exists():
    mia_model_dir.mkdir()
  
  for i in range(class_number):
    print(f"training the MIA model for class {i}")  
    optimizer = optim.Adam(mia_models[i].parameters(), **optim_args)  
                                                        
    train_loader = torch.utils.data.DataLoader(mia_train_datasets[i], batch_size = mia_batch_size, 
                                               shuffle = True, **cuda_args)    
                                                        
    balanced_test_dataset = BalancedSampler(mia_test_datasets[i])
    test_loader  = torch.utils.data.DataLoader(mia_test_datasets[i], batch_size = 1000, 
                                               sampler = balanced_test_dataset, **cuda_args)
                                               
    stats.new_train(name = f"MIA model {i}", label = "mia-model")                                                                          
    for epoch in range(mia_train_epochs):
      stats.new_epoch()
      train(mia_models[i].to(device), device, train_loader, optimizer, epoch, verbose = False)
      test(mia_models[i].to(device), device, test_loader, test_stats = stats)
    
    torch.save(mia_models[i], (mia_model_dir/f"class_{i}.pt").as_posix())
    
