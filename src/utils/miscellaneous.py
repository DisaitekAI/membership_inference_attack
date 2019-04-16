import datetime
import torch
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset

def progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
  """
  Call in a loop to create terminal progress bar
  @params:
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
  def __init__(self, dataset):
    self.indices     = list(range(len(dataset)))
    self.num_samples = len(self.indices)
    dataset_labels   = [y.item() for _, y in dataset] 
    label_counter    = Counter(dataset_labels)
    weights          = [1. / label_counter[label] for label in dataset_labels]
    self.weights     = torch.DoubleTensor(weights)

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
