import os, sys, inspect

project_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) + "/../../"
data_dir = project_dir + "/data/"

import torch
from torchvision import datasets, transforms

class Dataset_generator:
  """
  Generate datasets for the offline target model and for the 
  Shadow_swarm_trainer. For now it is ultra simple as we deal only with 
  MNIST. 
  
  examples
    get the mnist dataset
      dg = Dataset_generator(method = "academic", name = "mnist", train = True)
      dataset = dg.generate() # use it as a torch.utils.data.Dataset
  """
  def __init__(self, *opt_dict, **opt_args):
    """
    :method "academic"
    :name "mnist". The name of an academic dataset
    :train boolean. Get the trainning set or the testing set from an 
      academic dataset.
    """
    # Setting default attributes values
    self.train = True
    
    # Overriding attributes
    for dictionary in opt_dict:
      for key in dictionary:
        setattr(self, key, dictionary[key])
    for key in opt_args:
      setattr(self, key, opt_args[key])

  def generate(self):
    """
    Generate the dataset with the arguments given at initialization.
     
    Academic datasets are transformed using their own norm and std 
    values.
    
    Always return a torch.utils.data.Dataset object
    """
    if self.method == "academic":
      if self.name == "mnist":
        return datasets.MNIST(data_dir, train=self.train, download=True,
                              transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    
