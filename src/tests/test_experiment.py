import os, sys, inspect

project_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) + "/../../"
models_dir = project_dir + "/models/"
src_models_dir = project_dir + "/src/models/"
utils_dir = project_dir + "/src/utils/"

# add ../models/ into the path
if src_models_dir not in sys.path:
  sys.path.insert(0, src_models_dir)
  
# add ../utils/ into the path
if utils_dir not in sys.path:
  sys.path.insert(0, utils_dir)
  
from experiment import experiment
from utils_modules import Flatten

from collections import OrderedDict
import torch
import torch.nn as nn

def test_experiment_mnist_basic():
  target_path = models_dir + "/test_mnist_model.pt"
  mia_path = models_dir + "/test_mia_model.pt"
  
  experiment(academic_dataset = "mnist", 
             target_model_path = target_path,
             mia_model_path = mia_path)
  
  assert os.path.isfile(target_path)
  assert os.path.isfile(mia_path)
  
  if os.path.isfile(target_path):
    os.remove(target_path) 
  if os.path.isfile(mia_path):
    os.remove(mia_path) 
  
def test_experiment_mnist_custom():
  target_path = models_dir + "/test_mnist_model.pt"
  mia_path = models_dir + "/test_mia_model.pt"
  
  experiment(academic_dataset = "mnist", 
             target_model_path = target_path,
             mia_model_path = mia_path,
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
             custom_mia_model = OrderedDict([
               ('dense1', nn.Linear(20, 50)),
               ('tanh', nn.Tanh()),
               ('dense2', nn.Linear(50, 2)),
               ('logsoftmax', nn.LogSoftmax(dim=1))
             ]),
             custom_mia_optim_args = {'lr' : 0.02, 'momentum' : 0.3})
  
  assert os.path.isfile(target_path)
  assert os.path.isfile(mia_path)
  
  if os.path.isfile(target_path):
    os.remove(target_path) 
  if os.path.isfile(mia_path):
    os.remove(mia_path) 
  
def main():
  test_experiment_mnist_basic()
  test_experiment_mnist_custom()

if __name__ == "__main__":
  main()
