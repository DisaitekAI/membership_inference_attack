import pathlib

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
data_path = home_path/'data'

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
  def __init__(self, method = "academic", train = True, name = None):
    """
    :method "academic"
    :name "mnist". The name of an academic dataset
    :train boolean. Get the trainning set or the testing set from an 
      academic dataset.
    """
    self.method = method
    self.train  = train
    self.name   = name

  def generate(self):
    """
    Generate the dataset with the arguments given at initialization.
     
    Academic datasets are transformed using their own norm and std 
    values.
    
    Always return a torch.utils.data.Dataset object
    """
    if self.method == "academic":
      if self.name == "mnist":
        return datasets.MNIST(data_path.as_posix(), train=self.train, download=True,
                              transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    
