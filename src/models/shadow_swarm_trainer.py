import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
utils_path = home_path/'src'/'utils'

if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())

import torch
import torch.utils.data
import torch.optim as optim
from copy import deepcopy
from utils_modules import weight_init, train
from miscellaneous import progress_bar
from torch.utils.data import TensorDataset

def max_label(data_loader):
  """
  Super dirty function to get the maximum label from any dataset.
  Use it when there is no other choice.
  """
  m = 0
  for _, targets in data_loader:
    for target in targets:
      if target > m:
        m = target
        
  return m.item()

def get_mia_dataset(dataset                  = None, 
                    shadow_number            = None, 
                    shadow_model             = None, 
                    use_cuda                 = False,
                    custom_shadow_optim_args = None,
                    shadow_model_base_path   = None,
                    mia_dataset_path         = None):
  """
  create a dataset for the MIA model.
  
  :dataset 
  :shadow_number 
  :shadow_model
  :use_cuda
  :custom_shadow_optim_args
  :shadow_model_base_path
  """
  
  if (dataset                is None) or \
     (shadow_number          is None) or \
     (shadow_model           is None) or \
     (mia_dataset_path       is None) or \
     (shadow_model_base_path is None):
    raise ValueError("swhadow_swarm_trainer: the following arguments" 
                     "are required to be set: shadow_model_base_path"
                     ", dataset, shadow_number, shadow_model, "
                     "mia_dataset_path")
  
  # test if shadow models needs to be trained or whether the work is 
  # already done. So we test if there is a shadow model with the last 
  # index, meaning that shadow models have been trained. 
  # TODO(PI) deal with the case where the shadow model differs
  last_shadow_model_path = pathlib.Path(shadow_model_base_path + "_{}.pt".format(shadow_number - 1))
  more_shadow_model_path = pathlib.Path(shadow_model_base_path + "_{}.pt".format(shadow_number))
  
  cuda_args = { 'num_workers' : 1, 'pin_memory' : True } if use_cuda else {}
  device = torch.device('cuda' if use_cuda else 'cpu')
  dataset_size = len(dataset)
  shadow_models = []
  
  if last_shadow_model_path.exists() and (not more_shadow_model_path.exists()):
    if pathlib.Path(mia_dataset_path).exists():
      # all the work is done, return the last dataset
      print("\nloading the last MIA dataset")
      return torch.load(mia_dataset_path)
    else:
      # the MIA dataset creation has not been done, load the shadow models
      print("\nloading shadow models")
      for i in range(shadow_number):
        shadow_models.append(torch.load(shadow_model_base_path + "_{}.pt".format(i)))
  else:
    # nothing has been done, train the shadow models
    print("\ntraining shadow models")
    
    shadow_dir = pathlib.Path(shadow_model_base_path).parent
    if not shadow_dir.exists():
      shadow_dir.mkdir()
    
    base_dataset_size = int(dataset_size / shadow_number)
  
    # set the dataset size for (shadow_number-1) shadows
    shadow_datasets_sizes = []
    for i in range(shadow_number - 1):
      shadow_datasets_sizes.append(base_dataset_size)
      
    # last size is the remaining size for the last shadow
    shadow_datasets_sizes.append(dataset_size - (base_dataset_size * (shadow_number - 1))) 
    
    # split the dataset into almost equal sizes
    shadow_datasets = torch.utils.data.random_split(dataset, shadow_datasets_sizes)
    
    for i in range(shadow_number):
      # copy model parameters but we wanna keep the weights randomized
      # differently between shadows (initialized at training time, see later)
      model = deepcopy(shadow_model)
      shadow_models.append(model)
    
    # remove old models if necessary
    if more_shadow_model_path.exists():
      base_file      = pathlib.Path(shadow_model_base_path)
      base_dir       = base_file.parent
      base_file_name = base_file.name
      
      old_shadow_models = base_dir.glob(base_file_name + '_*.pt')
      for i in range(len(old_shadow_models)):
        old_shadow_models.unlink()
      
    # shadow swarm training
    for i in range(shadow_number):
      model = shadow_models[i]
      model.apply(weight_init)
      
      data_loader = torch.utils.data.DataLoader(shadow_datasets[i], batch_size = 32, 
                                                shuffle = True, **cuda_args)
      optim_args = { 'lr' : 0.01, 'momentum' : 0.5 }
      if custom_shadow_optim_args is not None:
        optim_args = custom_shadow_optim_args
          
      optimizer = optim.SGD(model.parameters(), **optim_args)
      
      for epoch in range(20):
        train(model, device, data_loader, optimizer, epoch, verbose = False)
      
      torch.save(model, shadow_model_base_path + "_{}.pt".format(i))
      progress_bar(iteration = i, total = shadow_number - 1)
  
  # set all shadow models in evaluation mode
  print("\nbuilding the MIA dataset")
  for i in range(shadow_number):
    shadow_models[i].eval()
    
  # build the MIA dataset
  mia_dataset                   = []
  input_activation_tensor_list  = list()
  input_targetclass_tensor_list = list()
  output_tensor_list            = list()
  
  i = 0
  test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, **cuda_args)
  class_number = max_label(test_loader) + 1 # assuming the target solves a classification task 
  
  # TODO(PI) no maximal parallelism right now because of a batch size of 1,
  # so it's quite long. The problem is we have to keep track of which 
  # sample belongs to which shadow training dataset. It requires to fix 
  # this first to enable full parallelism with batch size.
  with torch.no_grad():
    for data, targets in test_loader:
      data, targets = data.to(device), targets.to(device)
      
      one_hot_target = torch.zeros([1, class_number], dtype = torch.float)
      one_hot_target[0][targets[0]] = 1.0 # assuming the target solves a classification task 

      shadow_index = int(i/shadow_number)
      i += 1
      for j in range(shadow_number):
        mia_output = torch.tensor([0])
        if j == shadow_index:
          mia_output = torch.tensor([1])
          
        shadow_output = shadow_models[j](data)
        input_activation_tensor_list.append(shadow_output)
        input_targetclass_tensor_list.append(one_hot_target)
        output_tensor_list.append(mia_output)
        
      progress_bar(iteration = i, total = dataset_size)

  mia_dataset = TensorDataset(torch.cat(input_activation_tensor_list),
                              torch.cat(input_targetclass_tensor_list),
                              torch.cat(output_tensor_list))
  torch.save(mia_dataset, mia_dataset_path)
  
  return mia_dataset
      
    
    
    
  
