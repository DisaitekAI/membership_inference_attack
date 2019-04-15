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

def get_mia_train_dataset(dataset                  = None, 
                          shadow_number            = None, 
                          shadow_model             = None, 
                          use_cuda                 = False,
                          custom_shadow_optim_args = None,
                          shadow_model_base_path   = None,
                          mia_dataset_path         = None,
                          class_number             = None):
  """
  create a dataset for the MIA model.
  
  :dataset 
  :shadow_number 
  :shadow_model
  :use_cuda
  :custom_shadow_optim_args
  :shadow_model_base_path
  :mia_dataset_path
  :class_number
  """
  
  if (dataset                is None) or \
     (shadow_number          is None) or \
     (shadow_model           is None) or \
     (mia_dataset_path       is None) or \
     (class_number           is None) or \
     (shadow_model_base_path is None):
    raise ValueError("get_mia_train_dataset: one of the requiered "
                     "argument is not set")
  
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
  mia_datasets_dir = pathlib.Path(mia_dataset_path)
  
  if last_shadow_model_path.exists() and (not more_shadow_model_path.exists()):
    need_to_work = False
    for i in range(class_number):
      if not (mia_datasets_dir/f"class_{i}.pt").exists():
        # some of the training dataset is missing, so they have to be 
        # built
        need_to_work = True
        break
    
    if not need_to_work:
      print("loading the MIA train datasets")
      mia_datasets = list()
      for i in range(class_number):
        mia_datasets.append(torch.load((mia_datasets_dir/f"class_{i}.pt").as_posix()))
        
      return mia_datasets
      
    if not mia_datasets_dir.exists():
      mia_datasets_dir.mkdir()
      
    # the MIA dataset creation has not been done, load the shadow models
    print("\nloading shadow models")
    for i in range(shadow_number):
      shadow_models.append(torch.load(f"{shadow_model_base_path}_{i}.pt"))
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
      
      old_shadow_models = list(base_dir.glob(base_file_name + '_*.pt'))
      for i in range(len(old_shadow_models)):
        old_shadow_models[i].unlink()
      
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
  print("\nbuilding the MIA train datasets")
  for i in range(shadow_number):
    shadow_models[i].eval()
    
  # build the MIA datasets
  input_tensor_lists  = list()
  output_tensor_lists = list()
  for i in range(class_number):
    input_tensor_lists.append(list())
    output_tensor_lists.append(list())
  
  i = 0
  test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, **cuda_args)
  
  # TODO(PI) no maximal parallelism right now because of a batch size of 1,
  # so it's quite long. The problem is we have to keep track of which 
  # sample belongs to which shadow training dataset. It requires to fix 
  # this first to enable full parallelism with batch size.
  with torch.no_grad():
    for data, targets in test_loader:
      data, targets = data.to(device), targets.to(device)

      shadow_index = int(i/shadow_number)
      i += 1
      for j in range(shadow_number):
        mia_output = torch.tensor([0])
        if j == shadow_index:
          mia_output = torch.tensor([1])
          
        shadow_output = shadow_models[j](data)
        input_tensor_lists[targets[0]].append(shadow_output)
        output_tensor_lists[targets[0]].append(mia_output)
        
      progress_bar(iteration = i, total = dataset_size)
  
  mia_datasets = list()
  for i in range(class_number):
    mia_datasets.append(TensorDataset(torch.cat(input_tensor_lists[i]),
                                      torch.cat(output_tensor_lists[i])))

    torch.save(mia_datasets[i], (mia_datasets_dir/f"class_{i}.pt").as_posix())
  
  return mia_datasets
      
def get_mia_test_dataset(train_dataset    = None,
                         test_dataset     = None,
                         target_model     = None,
                         use_cuda         = False,
                         mia_dataset_path = None,
                         class_number     = None):
  if (mia_dataset_path is None) or (class_number is None):
    raise ValueError("get_mia_test_dataset: one of the required "
                     "argument is not set")
  
  mia_datasets_dir = pathlib.Path(mia_dataset_path)                   
  need_to_work = False
  for i in range(class_number):
    if not (mia_datasets_dir/f"class_{i}.pt").exists():
      # some of the training dataset is missing, so they have to be 
      # built
      need_to_work = True
      break
  
  if not need_to_work:
    print("loading the MIA test datasets")
    mia_datasets = list()
    for i in range(class_number):
      mia_datasets.append(torch.load((mia_datasets_dir/f"class_{i}.pt").as_posix()))
      
    return mia_datasets
  
  if not mia_datasets_dir.exists():
    mia_datasets_dir.mkdir()
      
  print('building the MIA test datasets')
                           
  cuda_args = { 'num_workers' : 1, 'pin_memory' : True } if use_cuda else {}
  device = torch.device('cuda' if use_cuda else 'cpu')  
  
  input_tensor_lists  = list()
  output_tensor_lists = list()
  for i in range(class_number):
    input_tensor_lists.append(list())
    output_tensor_lists.append(list())
  
  i = 0
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False, **cuda_args)
  test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, **cuda_args) 
  dataset_size = len(train_dataset) + len(test_dataset)
  
  with torch.no_grad():
    for data, targets in train_loader:
      i += 1
      data, targets = data.to(device), targets.to(device)
      
      output = target_model(data)
      input_tensor_lists[targets[0]].append(output)
      output_tensor_lists[targets[0]].append(torch.tensor([0]))
        
      progress_bar(iteration = i, total = dataset_size)
      
    for data, targets in test_loader:
      i += 1
      data, targets = data.to(device), targets.to(device)

      output = target_model(data)
      input_tensor_lists[targets[0]].append(output)
      output_tensor_lists[targets[0]].append(torch.tensor([0]))
        
      progress_bar(iteration = i, total = dataset_size)
      
  mia_datasets = list() 
  for i in range(class_number):
    mia_datasets.append(TensorDataset(torch.cat(input_tensor_lists[i]),
                                      torch.cat(output_tensor_lists[i])))
    torch.save(mia_datasets[i], (mia_datasets_dir/f"class_{i}.pt").as_posix())
  
  return mia_datasets
