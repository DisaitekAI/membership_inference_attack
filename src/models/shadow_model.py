import torch
import torch.utils.data
import mnist_model.py

shadow_data = torch.utils.data.Dataset
current_model = Mnist_model

M = 7
batch_size = 64

shadow_dataset_size = len(shadow_data) // M # suppose que ça tombe juste

unsplit_shadow_datasets = random_split(shadow_data, shadow_dataset_size)

split_shadow_datasets = []
for dataset in unsplit_shadow_datasets:
  learnable_shadow_dataset = random_split(dataset, batch_size)
  split_shadow_datasets.append(learnable_shadow_dataset)

shadow_models = []
for split_dataset in split_shadow_datasets:
  shadow_model = current_model(split_dataset) # pas sûr que ce soit la bonne syntaxe
  shadow_models.append(shadow_model)
