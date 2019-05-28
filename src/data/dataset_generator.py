import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
data_path = home_path/'data'
utils_path = home_path/'src'/'utils'

if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())

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
import pickle
import zipfile

from miscellaneous import fixed_random_split

class Dataset_generator:
  """generates datasets for the offline target model or the online 
  target model.
  
  Example:
    get the mnist dataset::
    
      dg = Dataset_generator(method = "academic", name = "mnist", train = True)
      dataset = dg.generate() # returns a torch.utils.data.Dataset
  
  """
  def __init__(self, method = 'academic', train = True, name = None):
    """
    Args:
      method (string): "academic".
      
      name (string): name of the academic dataset to use. Currently implemented values "mnist", "cifar10", "Federal".
      
      train (bool): whether to get the train or test academic dataset.
      
    """
    self.method = method
    self.train  = train
    self.name   = name
  
  def federal_process_columns(self, df_data, df_target):
    # encode and normalize columns

    numerical_columns = ['LOS']
    # Numerical column normalization
    df_num            = df_data[numerical_columns]
    num_val_mean      = df_num.mean(axis = 0)
    num_val_std       = df_num.std(axis = 0)
    df_num            = (df_num - num_val_mean) / num_val_std

    # Categorical columns encoding
    df_cat            = df_data.drop(numerical_columns, axis = 1)
    columns_encoders = {
      col : {
        val : i
        for i, val in enumerate(df_cat[col].unique())
      }
      for col in df_cat.columns
    }
    target_encoder = {
      val : i
      for i, val in enumerate(sorted(df_target.unique()))
    }
    column_order = sorted(columns_encoders.keys())

    for col in df_cat.columns:
      df_cat[col] = df_cat[col].apply(lambda x: columns_encoders[col][x])
    df_target = df_target.map(lambda x: target_encoder[x])

    return (df_cat, df_num, df_target, column_order, columns_encoders,
            target_encoder)

  def federal_create_pytorch_dataset(self, df_cat, df_num, df_target, column_order):
    dataset = TensorDataset(
      torch.tensor(df_num.values, dtype = torch.float32), # numerical variables
      *[
          torch.tensor(df_cat[col].values)
          for col in column_order
      ], # categorical variables in the correct order
      torch.tensor(df_target.values, dtype = torch.int64) # target variables
    )

    return dataset

  def federal_split_dataset(self, dataset, valid_prop = 0.2):
    dataset_size                 = len(dataset)
    valid_size                   = round(valid_prop * dataset_size)
    lengths                      = [dataset_size - valid_size, valid_size]
    train_dataset, valid_dataset = fixed_random_split(dataset, lengths)

    return train_dataset, valid_dataset
    
  def federal(self):
    # dowload the dataset if necessary
    if not (data_path/'Federal').exists():
      (data_path/'Federal').mkdir()
      
    if not (data_path/'Federal'/'processed').exists():
      (data_path/'Federal'/'processed').mkdir()
      
    if not (data_path/'Federal'/'raw').exists():
      (data_path/'Federal'/'raw').mkdir()
    
    unzipped_file_path = (data_path/'Federal'/'raw'/'FACTDATA_SEP2018.TXT')
    if not unzipped_file_path.exists():
      print('downloading the federal employee dataset')
      zipped_path = (data_path/'Federal'/'raw'/'9e7f077c-a17a-46b6-a79a-d61c55d39841.zip')
      
      url = 'https://www.opm.gov/Data/Files/549/9e7f077c-a17a-46b6-a79a-d61c55d39841.zip'
       
      urllib.request.urlretrieve(url, zipped_path.as_posix())  

      with zipfile.ZipFile(zipped_path.as_posix(), 'r') as zip_obj:
        listOfFileNames = zip_obj.namelist()
        for fileName in listOfFileNames:
          if fileName.endswith('.TXT'):
            zip_obj.extract(fileName, unzipped_file_path.parent.as_posix())
          
      zipped_path.unlink()

    # preprocessing
    # col_desc = {
    #   'AGYSUB'    : 'Agency',
    #   'LOC'       : 'Location',
    #   'AGELVL'    : 'Age (bucket)',
    #   'EDLVL'     : 'Education level',
    #   'GSEGRD'    : 'General schedule & Equivalent grade',
    #   'LOSLVL'    : 'Length of service (bucket)',
    #   'OCC'       : 'Occupation',
    #   'PATCO'     : 'Occupation category',
    #   'PPGRD'     : 'Pay Plan & Grade',
    #   'STEMOCC'   : 'STEM Occupation',
    #   'SUPERVIS'  : 'Supervisory status',
    #   'TOA'       : 'Type of appointment',
    #   'WORKSCH'   : 'Work schedule',
    #   'WORKSTAT'  : 'Work status',
    #   'LOS'       : 'Average length of service',
    #   'SALBUCKET' : 'Salary bucket'
    # }
    train_set = None
    test_set  = None
    if (not (data_path/'Federal'/'processed'/'train.pt').exists()) or (not (data_path/'Federal'/'processed'/'train.pt').exists()): 
      print('preprocessing of the federal employee dataset')
      df                                   = pd.read_csv(unzipped_file_path)
      df                                   = df.sample(100000, random_state = 42)
      df.EDLVL[df.EDLVL == '**']           = float('nan')
      df.EDLVL                             = df.EDLVL.astype(float)
      df.GSEGRD[df.GSEGRD == '**']         = float('nan')
      df.GSEGRD                            = df.GSEGRD.astype(float)
      df.OCC[df.OCC == '****']             = float('nan')
      df.OCC                               = df.OCC.astype(float)
      df.SUPERVIS[df.SUPERVIS == '*']      = float('nan')
      df.SUPERVIS                          = df.SUPERVIS.astype(float)
      df.TOA[df.TOA == '**']               = float('nan')
      df.TOA                               = df.TOA.astype(float)
      df                                   = df.drop(['DATECODE', 'EMPLOYMENT'], axis = 1)
  
      # Removing the nan values in columns by either adding a new category
      # or dropping the lines
      df                                   = df[~df.EDLVL.isnull()]
      df.loc[df.GSEGRD.isnull(), 'GSEGRD'] = 0
      df.loc[df.OCC.isnull(), 'OCC']       = 0
      df                                   = df[~df.SUPERVIS.isnull()]
      df                                   = df[~df.TOA.isnull()]
      df                                   = df[~df.SALARY.isnull()]
      df                                   = df[~df.LOS.isnull()]
      # Target generation, we partition the salary values in
      # `n_salary_slice` equally sized buckets.
      slice_size                           = 1 / 10.0
      df['SALBUCKET']                      = pd.qcut(
        df.SALARY,
        np.arange(
            0,
            1 + slice_size,
            slice_size
        )
      )

      # split inputs and outputs to predict
      df_data   = df.drop(['SALBUCKET', 'SALARY', 'SALLVL'], axis = 1)
      df_target = df['SALBUCKET']

      (
        df_cat,
        df_num,
        df_target,
        column_order,
        columns_encoders,
        target_encoder
      ) = self.federal_process_columns(df_data, df_target)

      dataset = self.federal_create_pytorch_dataset(
          df_cat,
          df_num,
          df_target,
          column_order
      )

      (
          train_set,
          test_set
      ) = self.federal_split_dataset(dataset)

      torch.save(train_set, (data_path/'Federal'/'processed'/'train.pt').as_posix())
      torch.save(test_set, (data_path/'Federal'/'processed'/'test.pt').as_posix())
      
      with open((data_path/'Federal'/'processed'/'meta.pt').as_posix(), 'wb') as pickle_file:
        pickle.dump([column_order, columns_encoders, target_encoder, df_num.shape[1]], pickle_file)
      
    else:
      if self.train:
        train_set = torch.load((data_path/'Federal'/'processed'/'train.pt').as_posix())
      else:
        test_set  = torch.load((data_path/'Federal'/'processed'/'test.pt').as_posix())

    if self.train:
      return train_set
    else:
      return test_set
    
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
    """Generate the dataset with the arguments given at initialization.
     
    Academic datasets are transformed using their own norm and std 
    values.
    
    Returns:
      dataset (torch Dataset): the requested dataset.
      
    """
    if self.method == 'academic':
      if self.name == 'mnist':
        return self.mnist()
    
      if self.name == 'purchase':
        return self.purchase()
        
      if self.name == 'cifar10':
        return self.cifar10()

      if self.name == 'federal':
        return self.federal()
    
