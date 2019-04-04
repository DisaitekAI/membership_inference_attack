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
    :offline boolean set True by default
    :model_path path to the trained model
    :device torch.device("cpu") by default
    """
    self.offline = offline
    self.device = device
    
    if offline == True:
      if pathlib.Path(model_path).exists():
        self.offline_model = torch.load(model_path)
        self.offline_model.eval()
        
      else:
        raise ValueError('Target: the model path is invalid')
      
  def predict(vinput):
    if self.offline:
      return self.offline_model(vinput).to(self.device)
      
       
