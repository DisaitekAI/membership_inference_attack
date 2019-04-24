import torch

def membership_distributions(datasets):
  print('Membership distributions')
  # iterate through attack model classes
  i = 0
  for dataset in datasets:
    s_in  = []
    s_out = []
    
    #iterate through the samples
    for s_input, s_output in dataset:
      if s_output == 1:
        s_in.append(s_input)
      else:
        s_out.append(s_input)
        
    s_in  = torch.exp(torch.stack(s_in))
    s_out = torch.exp(torch.stack(s_out))
    
    print(f"class {i}")
    print(f"  distribution of in samples: {s_in.mean(dim = 0)}")
    print(f"  distribution of out samples: {s_out.mean(dim = 0)}")
    
    i += 1
    
      
