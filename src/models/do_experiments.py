import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
models_path     = home_path/'models'
src_models_path = home_path/'src'/'models'
utils_path      = home_path/'src'/'utils'
data_path       = home_path/'data'

# add ../models/ into the path
if src_models_path.as_posix() not in sys.path:
  sys.path.insert(0, src_models_path.as_posix())
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
from experiment import experiment
from utils_modules import Flatten
from collections import OrderedDict
import torch.nn as nn
  
def main():
  # ~ experiment(academic_dataset       = 'mnist', 
             # ~ target_model_path      = (models_path/'mnist_model.pt').as_posix(),
             # ~ mia_model_path         = (models_path/'mia_model.pt').as_posix(),
             # ~ shadow_model_base_path = (models_path/'shadows'/'shadow').as_posix(),
             # ~ mia_dataset_path       = (data_path/'mia_dataset.pt').as_posix())
  
  experiment(academic_dataset    = 'mnist', 
             target_model_path   = (models_path/'mnist_model.pt').as_posix(),
             mia_model_path      = (models_path/'mia_model.pt').as_posix(),
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
             shadow_number            = 50,
             shadow_model_base_path   = (models_path/'shadows'/'shadow').as_posix(),
             mia_dataset_path         = (data_path/'mia_dataset.pt').as_posix())

if __name__ == '__main__':
  main()
