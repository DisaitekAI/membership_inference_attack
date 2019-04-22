import pathlib

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
data_path = home_path/'data'

import torch
from torchvision import datasets, transforms
import urllib.request
import gzip
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import torch
from torch.utils.data import TensorDataset

class Dataset_generator:
  """
  Generate datasets for the offline target model and for the 
  Shadow_swarm_trainer.
  
  examples
    get the mnist dataset
      dg = Dataset_generator(method = "academic", name = "mnist", train = True)
      dataset = dg.generate() # use it as a torch.utils.data.Dataset
  """
  def __init__(self, method = 'academic', train = True, name = None):
    """
    :method "academic"
    :name "mnist". The name of an academic dataset
    :train boolean. Get the trainning set or the testing set from an 
      academic dataset.
    """
    self.method = method
    self.train  = train
    self.name   = name
    
  def cifar10(self):
    return datasets.CIFAR10(data_path.as_posix(), train=self.train, download=True,
                            transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))
    
    
  def mnist(self):
    return datasets.MNIST(data_path.as_posix(), train=self.train, download=True,
                          transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ]))
    
  def purchase(self):
    # download the preprocessed data
    if not (data_path/'Purchase').exists():
      (data_path/'Purchase').mkdir()
      
    if not (data_path/'Purchase'/'processed').exists():
      (data_path/'Purchase'/'processed').mkdir()
      
    if not (data_path/'Purchase'/'raw').exists():
      (data_path/'Purchase'/'raw').mkdir()
    
    unzipped_file_path = (data_path/'Purchase'/'raw'/'all_features.csv')
    if not unzipped_file_path.exists():
      zipped_path = (data_path/'Purchase'/'raw'/'all_features.csv.gz')
      
      url = 'https://github.com/auduno/Kaggle-Acquire-Valued-Shoppers-Challenge/raw/master/features/train/all_features.csv.gz'
       
      urllib.request.urlretrieve(url, zipped_path.as_posix())  
      with gzip.open(zipped_path.as_posix(), 'rb') as f_in:
        with open(unzipped_file_path.as_posix(), 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
          
      zipped_path.unlink()
      
    # transform it to a TensorDataset
    train_path = (data_path/'Purchase'/'processed'/'train.pt')
    test_path  = (data_path/'Purchase'/'processed'/'test.pt')
    if (not train_path.exists()) or \
       (not test_path.exists()):
         
      raw_data = pd.io.parsers.read_csv(unzipped_file_path.as_posix(), sep = " ")
      rs = ShuffleSplit(n_splits = 1, test_size = 0.2)
      train_data, test_data = None, None
      for tr, te in rs.split(raw_data):
        train_data = raw_data.iloc[tr,:]
        test_data  = raw_data.iloc[te,:]
        
      train_label = train_data['label']
      test_label = test_data['label']
      
      # remove labels from the feature vectors
      del train_data['label']
      del train_data['repeattrips']
      del test_data['label']
      del test_data['repeattrips']

      # remove features id based features
      del train_data['offer_id']
      del test_data['offer_id']
      del train_data['chain']
      del test_data['chain']
      del train_data['id']
      del test_data['id']
      
      train_set = TensorDataset(torch.tensor(train_data.values).float(), 
                                torch.tensor(train_label.to_list()).long())
      test_set  = TensorDataset(torch.tensor(test_data.values).float(), 
                                torch.tensor(test_label.to_list()).long())
      
      torch.save(train_set, train_path.as_posix())
      torch.save(test_set, test_path.as_posix())
      
      if self.train:
        return train_set
      else:
        return test_set
      
    else:
      if self.train:
        return torch.load(train_path.as_posix())
      else:
        return torch.load(test_path.as_posix())

  def generate(self):
    """
    Generate the dataset with the arguments given at initialization.
     
    Academic datasets are transformed using their own norm and std 
    values.
    
    Always return a torch.utils.data.Dataset object
    """
    if self.method == 'academic':
      if self.name == 'mnist':
        return self.mnist()
    
      if self.name == 'purchase':
        return self.purchase()
        
      if self.name == 'cifar10':
        return self.cifar10()
        
          
          
        
          
        
          
        
    
