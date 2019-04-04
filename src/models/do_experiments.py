import pathlib

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
models_path = home_path/'models'
data_path   = home_path/'data'
  
from experiment import experiment
  
def main():
  experiment(academic_dataset       = 'mnist', 
             target_model_path      = (models_path/'mnist_model.pt').as_posix(),
             mia_model_path         = (models_path/'mia_model.pt').as_posix(),
             shadow_model_base_path = (models_path/'shadows'/'shadow').as_posix(),
             mia_dataset_path       = (data_path/'mia_dataset.pt').as_posix())

if __name__ == '__main__':
  main()
