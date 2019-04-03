import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
models_path = home_path/'models'
src_models_path = home_path/'src'/'models'
utils_path = home_path/'src'/'utils'

# add ../models/ into the path
if src_models_path.as_posix() not in sys.path:
  sys.path.insert(0, src_models_path.as_posix())
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
from experiment import experiment
from utils_modules import Flatten

from collections import OrderedDict
import torch
import torch.nn as nn

def test_experiment_mnist_basic():
  """
  Test a default MIA experiment on the MNIST dataset
  """
  target_path = models_path/'test_mnist_model.pt'
  mia_path = models_path/'test_mia_model.pt'
  
  experiment(academic_dataset  = "mnist", 
             target_model_path = target_path.as_posix(),
             mia_model_path    = mia_path.as_posix())
  
  assert target_path.exists()
  assert mia_path.exists()
  target_path.remove_p()
  mia_path.remove_p()
  
def test_experiment_mnist_custom():
  """
  Test of a MIA on the MNIST dataset with custom model for the MNIST
  model, custom mode for the MIA model and custom optimizer options
  """
  target_path = models_path/'test_mnist_model.pt'
  mia_path = models_path/'test_mia_model.pt'
  
  experiment(academic_dataset    = "mnist", 
             target_model_path   = target_path.as_posix(),
             mia_model_path      = mia_path.as_posix(),
             custom_target_model = OrderedDict([
               ('conv1', nn.Conv2d(1, 10, 3, 1)),
               ('relu1', nn.ReLU()),
               ('maxpool1', nn.MaxPool2d(2, 2)),
               ('conv2', nn.Conv2d(10, 10, 3, 1)),
               ('relu2', nn.ReLU()),
               ('maxpool2', nn.MaxPool2d(2, 2)),
               ('to1d', Flatten()),
               ('dense1', nn.Linear(11*11*10, 500)),
               ('tanh', nn.Tanh()),
               ('dense2', nn.Linear(500, 10)),
               ('logsoftmax', nn.LogSoftmax(dim=1))
             ]),
             custom_target_optim_args = {'lr' : 0.02, 'momentum' : 0.3},
             custom_mia_model         = OrderedDict([
               ('dense1', nn.Linear(20, 50)),
               ('tanh', nn.Tanh()),
               ('dense2', nn.Linear(50, 2)),
               ('logsoftmax', nn.LogSoftmax(dim=1))
             ]),
             custom_mia_optim_args = {'lr' : 0.02, 'momentum' : 0.3})
  
  assert target_path.exists()
  assert mia_path.exists()
  target_path.remove_p()
  mia_path.remove_p()
  
def main():
  test_experiment_mnist_basic()
  test_experiment_mnist_custom()

if __name__ == "__main__":
  main()
