import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
models_path = home_path/'models'
src_models_path = home_path/'src'/'models'
utils_path = home_path/'src'/'utils'
data_path = home_path/'data'

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
import pytest

target_path = models_path/'test_mnist_model.pt'
mia_path = models_path/'test_mia_model.pt'
shadow_base_path = models_path/'shadows'/'test_shadow'
mia_ds_path = data_path/'test_mia_dataset.pt'
  
@pytest.fixture(scope = 'function')
def remove_experiment_files():
  target_path.remove_p()
  mia_path.remove_p()
  mia_ds_path.remove_p
  
  base_dir = shadow_base_path.parent
  base_file_name = base_file.name
  
  old_shadow_models = list(base_dir.glob(base_file_name + '_*.pt'))
  for i in range(len(old_shadow_models)):
    old_shadow_models[i].unlink()

def test_experiment_mnist_basic():
  """
  Test a default MIA experiment on the MNIST dataset
  """
  experiment(academic_dataset       = 'mnist', 
             target_model_path      = target_path.as_posix(),
             mia_model_path         = mia_path.as_posix(),
             shadow_model_base_path = shadow_base_path.as_posix(),
             mia_dataset_path       = mia_ds_path.as_posix())
  
  assert target_path.exists()
  assert mia_path.exists()
  remove_experiment_files()
  
def test_experiment_mnist_custom(experiment_files_fixture):
  """
  Test of a MIA on the MNIST dataset with custom model for the MNIST
  model, custom mode for the MIA model and custom optimizer options
  """
  experiment(academic_dataset    = 'mnist', 
             target_model_path   = target_path.as_posix(),
             mia_model_path      = mia_path.as_posix(),
             custom_target_model = OrderedDict([
               ('conv1'       , nn.Conv2d(1, 10, 3, 1)),
               ('relu1'       , nn.ReLU()),
               ('maxpool1'    , nn.MaxPool2d(2, 2)),
               ('conv2'       , nn.Conv2d(10, 10, 3, 1)),
               ('relu2'       , nn.ReLU()),
               ('maxpool2'    , nn.MaxPool2d(2, 2)),
               ('to1d'        , Flatten()),
               ('dense1'      , nn.Linear(11*11*10, 500)),
               ('tanh'        , nn.Tanh()),
               ('dense2'      , nn.Linear(500, 10)),
               ('logsoftmax'  , nn.LogSoftmax(dim=1))
             ]),
             custom_target_optim_args = {'lr' : 0.02, 'momentum' : 0.3},
             custom_mia_model         = OrderedDict([
               ('dense1'      , nn.Linear(20, 50)),
               ('tanh'        , nn.Tanh()),
               ('dense2'      , nn.Linear(50, 2)),
               ('logsoftmax'  , nn.LogSoftmax(dim=1))
             ]),
             custom_mia_optim_args = {'lr' : 0.02, 'momentum' : 0.3},
             shadow_number         = 50,
             custom_shadow_model   = OrderedDict([
               ('conv1'       , nn.Conv2d(1, 15, 7, 1)),
               ('relu1'       , nn.ReLU()),
               ('maxpool1'    , nn.MaxPool2d(2, 2)),
               ('conv2'       , nn.Conv2d(15, 25, 7, 1)),
               ('relu2'       , nn.ReLU()),
               ('maxpool2'    , nn.MaxPool2d(2, 2)),
               ('to1d'        , Flatten()),
               ('dense1'      , nn.Linear(2*2*25, 50)),
               ('tanh'        , nn.Tanh()),
               ('dense2'      , nn.Linear(50, 10)),
               ('logsoftmax'  , nn.LogSoftmax(dim=1))
             ]),
             custom_shadow_optim_args = {'lr' : 0.02, 'momentum' : 0.3},
             shadow_model_base_path   = shadow_base_path.as_posix(),
             mia_dataset_path         = mia_ds_path.as_posix())
  
  assert target_path.exists()
  assert mia_path.exists()
  remove_experiment_files()
  
def main():
  test_experiment_mnist_basic()
  test_experiment_mnist_custom()

if __name__ == '__main__':
  main()
