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
from utils_modules import weight_init, train, test
from miscellaneous import progress_bar, fixed_random_split
from torch.utils.data import TensorDataset

def split_shadow_dataset(dataset, shadow_number):
  dataset_size = len(dataset)
  base_dataset_size = dataset_size // shadow_number
  
  # set the dataset size for (shadow_number-1) shadows
  shadow_datasets_sizes = []
  for i in range(shadow_number - 1):
    shadow_datasets_sizes.append(base_dataset_size)
    
  # last size is the remaining size for the last shadow
  shadow_datasets_sizes.append(dataset_size - (base_dataset_size * (shadow_number - 1))) 
  
  # split the dataset into almost equal sizes
  shadow_datasets_in = fixed_random_split(dataset, shadow_datasets_sizes)
  
  # the out samples are taken from the shadow number i-1 
  shadow_datasets_out = [shadow_datasets_in[0]]
  for i in range(1, shadow_number):
    shadow_datasets_out.append(shadow_datasets_in[i-1])
  
  return shadow_datasets_in, shadow_datasets_out
  

def get_mia_train_dataset(dataset                  = None, 
                          shadow_number            = None, 
                          shadow_model             = None, 
                          use_cuda                 = False,
                          custom_shadow_optim_args = None,
                          shadow_model_base_path   = None,
                          mia_dataset_path         = None,
                          class_number             = None,
                          stats                    = None,
                          shadow_train_epochs      = None,
                          shadow_batch_size        = None):
  """get_mia_train_dataset() create training datasets for the MIA models. 
  
  First it trains shadow models, which are usually copies of the target 
  model, then samples of the provided dataset are used to generate MIA 
  in and out sample by using the trained shadow models.
  
  Args:
    dataset (torch Dataset): a dataset for training the shadow models and generate in and out samples.                 
    
    shadow_number (int): the number of shadow models.          
    
    shadow_model (torch Module): the shadow model to use.            
    
    use_cuda (bool): whether to use cuda or not. False by default.                
    
    custom_shadow_optim_args (Dict): custom options for the Adam optimizer.
    
    shadow_model_base_path (string/Path): path of the shadow model dir. 
    
    mia_dataset_path (string/Path): path of the MIA training dataset dir.      
    
    class_number (int): number of classes in the problem solved by the target model.           
    
    stats (Statistics): statistics object that records the data of the shadow training.                   
    
    shadow_train_epochs (int): number of epoch for the shadow training.     
    
    shadow_batch_size (int): batch size of the shadow training.
    
  Returns:
    list(TensorDataset): a list of datasets to train the MIA model.
          
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
  
  cuda_args = { 'num_workers' : 1, 'pin_memory' : False } if use_cuda else {}
  device = torch.device('cuda' if use_cuda else 'cpu')
  dataset_size = len(dataset)
  shadow_models = []
  
  mia_datasets_dir = pathlib.Path(mia_dataset_path)
  if not mia_datasets_dir.exists():
    mia_datasets_dir.mkdir()
      
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
      
    shadow_datasets, _ = split_shadow_dataset(dataset, shadow_number)

    for i in range(shadow_number):
      # copy model parameters but we wanna keep the weights randomized
      # differently between shadows (initialized at training time, see later)
      model = deepcopy(shadow_model).to(device)
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
      model = shadow_models[i].to(device)
      model.apply(weight_init)
      
      j = 0
      if j == i:
        j = 1
      
      train_loader = torch.utils.data.DataLoader(shadow_datasets[i], batch_size = shadow_batch_size, 
                                                 shuffle = True, **cuda_args)
      test_loader = torch.utils.data.DataLoader(shadow_datasets[j], batch_size = 1000, 
                                                shuffle = True, **cuda_args)
                                                
      optim_args = {}
      if custom_shadow_optim_args is not None:
        optim_args = custom_shadow_optim_args
          
      optimizer = optim.Adam(model.parameters(), **optim_args)
      
      stats.new_train(label = "shadow-model")
      for epoch in range(shadow_train_epochs):
        train(model.to(device), device, train_loader, optimizer, epoch, verbose = False, train_stats = stats)
        if epoch == shadow_train_epochs - 1:
          stats.new_epoch()
          test(model.to(device), device, test_loader, test_stats = stats, verbose = False)
        
      torch.save(model, shadow_model_base_path + "_{}.pt".format(i))
      progress_bar(iteration = i, total = shadow_number - 1)
  
  # set all shadow models in evaluation mode
  print("\nbuilding the MIA train datasets")
  
  shadow_datasets_in, shadow_datasets_out = split_shadow_dataset(dataset, shadow_number)
  
  # build the MIA datasets
  input_tensor_lists  = [list() for i in range(class_number)]
  output_tensor_lists = [list() for i in range(class_number)]
  
  for i in range(shadow_number):
    current_shadow = shadow_models[i]
    current_shadow.eval()
    
    data_in_loader  = torch.utils.data.DataLoader(shadow_datasets_in[i], 
                       batch_size = 1000, shuffle = True, **cuda_args)
    
    with torch.no_grad():
      for batch in data_in_loader:
        data = batch[0:-1]
        targets = batch[-1]
        data = [e.to(device) for e in data]
        targets = targets.to(device)

        outputs = current_shadow(*data)
        
        for target, output in zip(targets, outputs):
          input_tensor_lists[target].append(output)
          output_tensor_lists[target].append(torch.tensor(1))
    
    data_out_loader = torch.utils.data.DataLoader(shadow_datasets_out[i], 
                       batch_size = 1000, shuffle = True, **cuda_args)
                       
    with torch.no_grad():
      for batch in data_out_loader:
        data = batch[0:-1]
        targets = batch[-1]
        data = [e.to(device) for e in data]
        targets = targets.to(device)
        
        outputs = current_shadow(*data)
        
        for target, output in zip(targets, outputs):
          input_tensor_lists[target].append(output)
          output_tensor_lists[target].append(torch.tensor(0))
  
  i = 0
  mia_datasets = list()
  for inputs, labels in zip(input_tensor_lists, output_tensor_lists):
    mia_datasets.append(TensorDataset(torch.stack(inputs), torch.stack(labels)))
    torch.save(mia_datasets[-1], (mia_datasets_dir/f"class_{i}.pt").as_posix())
    i += 1
  
  return mia_datasets


def get_mia_test_dataset(train_dataset    = None,
                         test_dataset     = None,
                         target_model     = None,
                         use_cuda         = False,
                         mia_dataset_path = None,
                         class_number     = None):
  """get_mia_test_dataset() generate test datasets for the MIA models.
  
  It executes, with the target model, samples coming from the train and 
  test dataset of the target model. 
  
  Args:
    train_dataset (torch Dataset): dataset used to train the target model.   
    
    test_dataset (torch Dataset): dataset that was not used to train the target model.   
    
    target_model (torch Module): the trained target model.    
    
    use_cuda (bool): whether to use cuda or not. False by default.       
    
    mia_dataset_path (string/Path): path for saving the MIA test datasets.
    
    class_number (int): number of classes in the problem solved by the target model.  
  
  Returns:
    list(TensorDataset): a list of datasets to test the MIA model.
  
  """
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
                           
  cuda_args = { 'num_workers' : 1, 'pin_memory' : False } if use_cuda else {}
  device = torch.device('cuda' if use_cuda else 'cpu')  
  
  input_tensor_lists  = [list() for i in range(class_number)]
  output_tensor_lists = [list() for i in range(class_number)]

  data_in_loader  = torch.utils.data.DataLoader(train_dataset, batch_size = 1000, 
                                                shuffle = True, **cuda_args)
  
  with torch.no_grad():
    for batch in data_in_loader:
      data = batch[0:-1]
      targets = batch[-1]
      data = [e.to(device) for e in data]
      targets = targets.to(device)
      
      outputs = target_model(*data)
      
      for target, output in zip(targets, outputs):
        input_tensor_lists[target].append(output)
        output_tensor_lists[target].append(torch.tensor(1))
  
  data_out_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1000, 
                                                shuffle = True, **cuda_args)
                     
  with torch.no_grad():
    for batch in data_out_loader:
      data = batch[0:-1]
      targets = batch[-1]
      data = [e.to(device) for e in data]
      targets = targets.to(device)
      
      outputs = target_model(*data)
      
      for target, output in zip(targets, outputs):
        input_tensor_lists[target].append(output)
        output_tensor_lists[target].append(torch.tensor(0))
  
  i = 0
  mia_datasets = list()
  for inputs, labels in zip(input_tensor_lists, output_tensor_lists):
    mia_datasets.append(TensorDataset(torch.stack(inputs), torch.stack(labels)))
    torch.save(mia_datasets[-1], (mia_datasets_dir/f"class_{i}.pt").as_posix())
    i += 1
  
  return mia_datasets
