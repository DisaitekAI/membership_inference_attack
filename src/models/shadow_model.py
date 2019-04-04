import torch
import torch.utils.data
import mnist_model
import dataset_generator

class swarm_shadow_model:
  def __init__(self, dataset, model, M):
    self.data = dataset
    self.model = model
    self.swarm_number = M
    self.shadow_size = len(dataset) // M
  def split(self):
    return random_split(self.data, self.shadow_size)
  def get_loader(self, batch_size):
    return [torch.utils.data.DataLoader(shadow_data, batch_size = batch_size, shuffle = True, **cuda_args) for shadow_data in split(self)]
  def training(self, batch_size, epoch_number):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    device = torch.device("cuda" if use_cuda else "cpu")
    shadow_models = [self.model for i in range(self.swarm_number)]
    loaders = get_loader(self, batch_size)
    index = 0
    for shadow in shadow_models:
      for epoch in range(epoch_number):
          train(shadow, device, loaders[index], optimizer, epoch)
      index += 1
    return shadow_models
  def generate_attack_data(self, batch_size, epoch_number):
    features, labels = self.data
    shadow_datasets = split(self)
    shadow_models = training(self, batch_size, epoch_number)
    attack_data = Dataset_generator(method = "academic", name = academic_dataset, train = False)
    for i in range(self.swarm_number):
      output = shadow_models[i](features)
      is_known = [k // (len(self.data) // self.swarm_number) == i for k in range(len(self.data))]
      current_shadow_attack_data = output + labels + is_known # ne convient pas, il faut concaténer selon l'axe des features...
      attack_data += current_shadow_attack_data
    return attack_data
