import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
models_path     = home_path/'models'
src_models_path = home_path/'src'/'models'
utils_path      = home_path/'src'/'utils'
data_path       = home_path/'data'
reports_path    = home_path/'reports'

# add ../models/ into the path
if src_models_path.as_posix() not in sys.path:
  sys.path.insert(0, src_models_path.as_posix())
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
from experiment import experiment
from utils_modules import Flatten, Print
from collections import OrderedDict
from statistics import Statistics
import torch.nn as nn
  
def main():
  # run the code on cuda or not for all experiments
  cuda = False
  if cuda:
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force = 'True')
  
  exp_stats = Statistics()
  
  for i in range(2, 130, 4):
    params = { 'academic_dataset'       : 'cifar10', 
               'target_model_path'      : (models_path/'cifar10_model_default.pt').as_posix(),
               'mia_model_path'         : (models_path/'mia_model_cifar10_default').as_posix(),
               'shadow_model_base_path' : (models_path/'shadows'/'shadow_cifar10_default').as_posix(),
               'mia_train_dataset_path' : (data_path/'mia_train_dataset_cifar10_default').as_posix(),
               'mia_test_dataset_path'  : (data_path/'mia_test_dataset_cifar10_default').as_posix(),
               'class_number'           : 10,
               'target_train_epochs'    : 15,
               'shadow_train_epochs'    : 15,
               'shadow_number'          : 90,
               'custom_mia_model'       : OrderedDict([
                 ('dense1'      , nn.Linear(10, 128)),
                 ('relu1'       , nn.ReLU()),
                 ('dropout1'    , nn.Dropout(0.3)),
                 ('dense2'      , nn.Linear(128, 64)),
                 ('relu2'       , nn.ReLU()),
                 ('dropout2'    , nn.Dropout(0.2)),
                 ('dense3'      , nn.Linear(64, 2)),
                 ('relu3'       , nn.ReLU()),
                 ('logsoftmax'  , nn.LogSoftmax(dim=1))
               ]),
               'custom_target_model'     : OrderedDict([
                 ('conv1', nn.Conv2d(3, 32, 3, 1)),
                 ('relu1', nn.ReLU()),
                 ('maxp1', nn.MaxPool2d(2, 2)),
                 ('conv2', nn.Conv2d(32, 64, 3, 1)),
                 ('relu2', nn.ReLU()),
                 ('maxp2', nn.MaxPool2d(2, 2)),
                 ('flatt', Flatten()),
                 ('dens1', nn.Linear(6*6*64, 512)),
                 ('relu3', nn.ReLU()),
                 ('dens2', nn.Linear(512, 10)),
                 ('lsoft', nn.LogSoftmax(dim=1))
               ]),
               'custom_shadow_model'     : OrderedDict([
                 ('conv1', nn.Conv2d(3, i, 3, 1)),
                 ('relu1', nn.ReLU()),
                 ('maxp1', nn.MaxPool2d(2, 2)),
                 ('conv2', nn.Conv2d(i, i, 3, 1)),
                 ('relu2', nn.ReLU()),
                 ('maxp2', nn.MaxPool2d(2, 2)),
                 ('flatt', Flatten()),
                 ('dens1', nn.Linear(6*6*i, 512)),
                 ('relu3', nn.ReLU()),
                 ('dens2', nn.Linear(512, 10)),
                 ('lsoft', nn.LogSoftmax(dim=1))
               ]),
               'use_cuda'                   : cuda,
               'no_mia_train_dataset_cache' : True,
               'no_mia_models_cache'        : True,
               'no_shadow_cache'            : True }
  
    exp_stats.new_experiment(f"Cifar10 MIA: shadow dense 1 neuron number {i} (vs target model: 512)", params)
    experiment(**params, stats = exp_stats)
    
  
  # ~ # default regularized purchase model
  # ~ params = { 'academic_dataset'       : 'purchase', 
             # ~ 'target_model_path'      : (models_path/'purchase_model_default.pt').as_posix(),
             # ~ 'mia_model_path'         : (models_path/'mia_model_purchase_default').as_posix(),
             # ~ 'shadow_model_base_path' : (models_path/'shadows'/'shadow_purchase_default').as_posix(),
             # ~ 'mia_train_dataset_path' : (data_path/'mia_train_dataset_purchase_default').as_posix(),
             # ~ 'mia_test_dataset_path'  : (data_path/'mia_test_dataset_purchase_default').as_posix(),
             # ~ 'class_number'           : 2,
             # ~ 'use_cuda'               : cuda }
  
  # ~ exp_stats.new_experiment("MIA on default Purchase model (batch norm + dropout regularization)", params)
  # ~ experiment(**params, stats = exp_stats)
  
  # ~ # default regularized mnist model
  # ~ params = { 'academic_dataset'       : 'mnist', 
             # ~ 'target_model_path'      : (models_path/'mnist_model_default.pt').as_posix(),
             # ~ 'mia_model_path'         : (models_path/'mia_model_default').as_posix(),
             # ~ 'shadow_model_base_path' : (models_path/'shadows'/'shadow_default').as_posix(),
             # ~ 'mia_train_dataset_path' : (data_path/'mia_train_dataset_default').as_posix(),
             # ~ 'mia_test_dataset_path'  : (data_path/'mia_test_dataset_default').as_posix(),
             # ~ 'class_number'           : 10,
             # ~ 'use_cuda'               : cuda }
  
  # ~ exp_stats.new_experiment("MIA on default Mnist (batch norm regularization)", params)
  # ~ experiment(**params, stats = exp_stats)
    
  # ~ # without regularization
  # ~ params = { 'academic_dataset'    : 'mnist', 
             # ~ 'target_model_path'   : (models_path/'mnist_model_exp1.pt').as_posix(),
             # ~ 'mia_model_path'      : (models_path/'mia_model_exp1').as_posix(),
             # ~ 'custom_target_model' : OrderedDict([
               # ~ ('conv1'       , nn.Conv2d(1, 10, 3, 1)),
               # ~ ('relu1'       , nn.ReLU()),
               # ~ ('maxpool1'    , nn.MaxPool2d(2, 2)),
               # ~ ('conv2'       , nn.Conv2d(10, 10, 3, 1)),
               # ~ ('relu2'       , nn.ReLU()),
               # ~ ('maxpool2'    , nn.MaxPool2d(2, 2)),
               # ~ ('to1d'        , Flatten()),
               # ~ ('dense1'      , nn.Linear(5*5*10, 500)),
               # ~ ('tanh'        , nn.Tanh()),
               # ~ ('dense2'      , nn.Linear(500, 10)),
               # ~ ('logsoftmax'  , nn.LogSoftmax(dim=1))
             # ~ ]),
             # ~ 'shadow_number'            : 50,
             # ~ 'shadow_model_base_path'   : (models_path/'shadows'/'shadow_exp1').as_posix(),
             # ~ 'mia_train_dataset_path'   : (data_path/'mia_train_dataset_exp1').as_posix(),
             # ~ 'mia_test_dataset_path'    : (data_path/'mia_test_dataset_exp1').as_posix(),
             # ~ 'class_number'             : 10,
             # ~ 'use_cuda'                 : cuda }
  
  # ~ exp_stats.new_experiment("MIA on Mnist with no regularization", params)
  # ~ experiment(**params, stats = exp_stats)
  
  # ~ # with dropout regularization
  # ~ params = { 'academic_dataset'    : 'mnist', 
             # ~ 'target_model_path'   : (models_path/'mnist_model_exp2.pt').as_posix(),
             # ~ 'mia_model_path'      : (models_path/'mia_model_exp2').as_posix(),
             # ~ 'custom_target_model' : OrderedDict([
               # ~ ('conv1'       , nn.Conv2d(1, 10, 3, 1)),
               # ~ ('relu1'       , nn.ReLU()),
               # ~ ('maxpool1'    , nn.MaxPool2d(2, 2)),
               # ~ ('dropout1'    , nn.Dropout(p = 0.5)),
               # ~ ('conv2'       , nn.Conv2d(10, 10, 3, 1)),
               # ~ ('relu2'       , nn.ReLU()),
               # ~ ('maxpool2'    , nn.MaxPool2d(2, 2)),
               # ~ ('dropout2'    , nn.Dropout(p = 0.5)),
               # ~ ('to1d'        , Flatten()),
               # ~ ('dense1'      , nn.Linear(5*5*10, 500)),
               # ~ ('tanh'        , nn.Tanh()),
               # ~ ('dropout3'    , nn.Dropout(p = 0.5)),
               # ~ ('dense2'      , nn.Linear(500, 10)),
               # ~ ('logsoftmax'  , nn.LogSoftmax(dim=1))
             # ~ ]),
             # ~ 'shadow_number'            : 50,
             # ~ 'shadow_model_base_path'   : (models_path/'shadows'/'shadow_exp2').as_posix(),
             # ~ 'mia_train_dataset_path'   : (data_path/'mia_train_dataset_exp2').as_posix(),
             # ~ 'mia_test_dataset_path'    : (data_path/'mia_test_dataset_exp2').as_posix(),
             # ~ 'class_number'             : 10,
             # ~ 'use_cuda'                 : cuda }
             
  # ~ exp_stats.new_experiment("MIA on Mnist with dropout regulrization", params)
  # ~ experiment(**params, stats = exp_stats)
             
  exp_stats.print_results()
  exp_stats.save(log_dir = reports_path)

if __name__ == '__main__':
  main()
