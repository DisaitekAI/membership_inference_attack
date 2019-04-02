import os, sys, inspect

project_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) + "/../../"
models_dir = project_dir + "/models/"
  
from experiment import experiment
  
def main():
  experiment(academic_dataset = "mnist", 
             target_model_path = models_dir + "/mnist_model.pt",
             mia_model_path = models_dir + "/mia_model.pt")

if __name__ == "__main__":
  main()
