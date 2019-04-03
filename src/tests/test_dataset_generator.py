import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
data_path = home_path/'data'
src_data_path = home_path/'src'/'data'

# add ../data/ into the path
if src_data_path.as_posix() not in sys.path:
  sys.path.insert(0, src_data_path.as_posix())

from dataset_generator import Dataset_generator
import torch

def test_dataset_generator_mnist():
  dg = Dataset_generator(method = "academic", name = "mnist", train = True)
  dataset = dg.generate()
  assert isinstance(dataset, torch.utils.data.Dataset)
  assert data_path/'MNIST'/'processed'/'training.pt'.exists()
  
  dg = Dataset_generator(method = "academic", name = "mnist", train = False)
  dataset = dg.generate()
  assert isinstance(dataset, torch.utils.data.Dataset)
  assert data_path/'MNIST'/'processed'/'test.pt'.exists()

def main():
  test_dataset_generator_mnist()

if __name__ == "__main__":
  main()
