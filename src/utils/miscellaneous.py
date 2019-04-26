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

def max_label(data_loader):
  """
  Super dirty function to get the maximum label from any dataset.
  Use it when there is no other choice.
  """
  m = 0
  for _, targets in data_loader:
    for target in targets:
      if target > m:
        m = target
        
  return m.item()
  
def fixed_random_split(dataset, lengths):
  """
  Randomly split a dataset into non-overlapping new datasets of given 
  lengths. The seed of this random function is always 42.

  Arguments:
      dataset (Dataset): Dataset to be split
      lengths (sequence): lengths of splits to be produced
  """
  if sum(lengths) != len(dataset):
    raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
  
  torch.manual_seed(42)
  indices = randperm(sum(lengths))
  torch.manual_seed(datetime.datetime.now().timestamp())
  
  return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
  
class BalancedSampler(torch.utils.data.sampler.Sampler):
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
