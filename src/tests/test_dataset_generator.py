import os, sys, inspect
  
# add ../data/ into the path
own_dir = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
data_dir = own_dir + "/../data/"
if data_dir not in sys.path:
  sys.path.insert(0, data_dir)

from dataset_generator import Dataset_generator
import torch

def test_dataset_generator_mnist():
  dg = Dataset_generator(method = "academic", name = "mnist", train = True)
  dataset = dg.generate()
  assert isinstance(dataset, torch.utils.data.Dataset)
  assert os.path.isfile("../../data/MNIST/processed/training.pt")
  
  dg = Dataset_generator(method = "academic", name = "mnist", train = False)
  dataset = dg.generate()
  assert isinstance(dataset, torch.utils.data.Dataset)
  assert os.path.isfile("../../data/MNIST/processed/test.pt")

def main():
  test_dataset_generator_mnist()

if __name__ == "__main__":
  main()
