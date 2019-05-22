import pathlib, sys
import torch

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
data_src_path = home_path/'src'/'data'
utils_path    = home_path/'src'/'utils'
viz_path      = home_path/'src'/'visualization'
model_path    = home_path/'src'/'models'
  
if utils_path.as_posix() not in sys.path:
  sys.path.insert(0, utils_path.as_posix())
  
if data_src_path.as_posix() not in sys.path:
  sys.path.insert(0, data_src_path.as_posix())

if viz_path.as_posix() not in sys.path:
  sys.path.insert(0, viz_path.as_posix())

if model_path.as_posix() not in sys.path:
  sys.path.insert(0, model_path.as_posix())

model = torch.load(sys.argv[1])

for param in model.parameters():
  print(param.data)

