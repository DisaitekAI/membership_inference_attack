from mnist_model import Mnist_model

import pathlib
import torch

class Target:
  """
  Class to interact with the target ML whether it is an offline model 
  that is trained or an online model that is requested
  """
  def __init__(self, offline = True, device = torch.device('cpu'), 
               model_path = None):
    """
    Args:
      offline (bool): whether we use an offline model or attack an online model. Set True by default.
    
      model_path path (string/Path): path for loading the target model.
    
      device (torch Device): torch.device("cpu") by default.
      
    """
    self.offline = offline
    self.device = device
    
    if offline == True:
      if pathlib.Path(model_path).exists():
        self.offline_model = torch.load(model_path)
        self.offline_model.eval()
        
      else:
        raise ValueError('Target: the model path is invalid')
      
  def __call__(self, vinput):
    """() execute a input sample with the target model.
    
    Args:
      vinput (torch Tensor): input for the model.
      
    Returns:
      output (torch Tensor): output of the model.
    """
    if self.offline:
      return self.offline_model(vinput).to(self.device)
      
       
