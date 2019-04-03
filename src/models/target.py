from mnist_model import Mnist_model

import pathlib
import torch

class Target:
  """
  Class to interact with the target ML whether it is an offline model 
  that is trained or an online model that is requested
  """
  def __init__(self, *opt_dict, **opt_args):
    """
    :offline boolean set True by default
    :model_path path to the trained model
    :device torch.device("cpu") by default
    """
    self.offline = True
    self.device = torch.device("cpu")
    
    for dictionary in opt_dict:
      for key in dictionary:
        setattr(self, key, dictionary[key])
    for key in opt_args:
      setattr(self, key, opt_args[key])
    
    if self.offline == True:
      if pathlib.Path(self.model_path).exists():
        self.offline_model = torch.load(self.model_path)
        self.offline_model.eval()
        
      else:
        raise ValueError("Target: the model path is invalid")
      
  def predict(vinput):
    if self.offline:
      return self.offline_model(vinput).to(self.device)
      
       
