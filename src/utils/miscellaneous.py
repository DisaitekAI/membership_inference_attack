import datetime
import torch
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset
from collections import Counter

def progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
  """
  Call in a loop to create terminal progress bar
  params:
  iteration   - Required  : current iteration (Int)
  total       - Required  : total iterations (Int)
  prefix      - Optional  : prefix string (Str)
  suffix      - Optional  : suffix string (Str)
  decimals    - Optional  : positive number of decimals in percent complete (Int)
  length      - Optional  : character length of bar (Int)
  fill        - Optional  : bar fill character (Str)
  """
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
  
  # new line on complete
  if iteration == total: 
    print()
  
def fixed_random_split(dataset, lengths):
  """Randomly split a dataset into non-overlapping new datasets of given 
  lengths. The seed of this random function is always 42.

  Args:
    dataset (torch Dataset): dataset to be split.
    lengths (list(int)): lengths of splits to be produced
  """
  if sum(lengths) != len(dataset):
    raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
  
  torch.manual_seed(42)
  indices = randperm(sum(lengths))
  torch.manual_seed(datetime.datetime.now().timestamp())
  
  return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
  
def fixed_random_subset(dataset, length, seed = 0):
  """Randomly subset a dataset with a given seed

  Args:
    dataset (torch Dataset): dataset to be split.
    length (int): length of the subset
    seed (int): seed for the random sample picking
  """
  if length > len(dataset):
    length = len(dataset) 
  
  torch.manual_seed(seed)
  indices = randperm(len(dataset))
  torch.manual_seed(datetime.datetime.now().timestamp())
  
  return Subset(dataset, indices[0:length])
  
class BalancedSampler(torch.utils.data.sampler.Sampler):
  """Sampler that draws around the same amount of sample for each class.
  """
  def __init__(self, dataset, oversampling = False):
    self.indices     = list(range(len(dataset)))
    dataset_labels   = [y.item() for _, y in dataset] 
    label_counter    = Counter(dataset_labels)
    weights          = [1. / label_counter[label] for label in dataset_labels]
    self.weights     = torch.DoubleTensor(weights)
    
    # ~ the data loader refuse to use more samples than the dataset size
    # ~ so this is not working. The sample number has the right value but the
    # ~ torch data loader refuse to use it.
    if oversampling:
      max_key = 0
      max_val = 0
      for key, val in label_counter.items():
        if max_key < key:
          max_key = key
        if max_val < val:
          max_val = val
  
      self.num_samples = max_val * (max_key + 1)
    else:
      self.num_samples = len(self.indices)

  def __iter__(self):
    return (
        self.indices[i] for i in torch.multinomial(
          self.weights,
          self.num_samples,
          replacement = True
      )
    )
    
  def __len__(self):
    return self.num_samples
