import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import pickle

home_path = pathlib.Path('.').resolve()
while home_path.name != "membership_inference_attack":
  home_path = home_path.parent
  
data_path = home_path/'data'

class CategoricalEmbeddings(nn.Module):
  def __init__(self, col_order, col_encoders, col_to_emb_dim):
    super(CategoricalEmbeddings, self).__init__()
    self.col_order = col_order
    self.cat_embs  = nn.ModuleDict({
      col: nn.Embedding(len(col_encoders[col]), col_to_emb_dim[col])
      for col in col_order
    })

  def forward(self, cat_variables):
    cat_variables = dict(zip(self.col_order, cat_variables))
    embeddings = [self.cat_embs[col](cat_variables[col]) for col in self.col_order]
    return torch.cat(embeddings, dim = 1)

class FederalModel(nn.Module):
  def __init__(self):
    super(FederalModel, self).__init__()
    meta_path = data_path/'Federal'/'processed'/'meta.pt'
    
    with open(meta_path, 'rb') as pickle_file:
      column_order, columns_encoders, target_encoder, num_var_number = pickle.load(pickle_file)
      
    linear_size = 256
    dropout_rate = 0.
    emb_dim = 5
    
    col_to_emb_dim  = { col : emb_dim for col in columns_encoders }
    class_number    = len(target_encoder)

    self.cat_emb    = CategoricalEmbeddings(column_order, columns_encoders, col_to_emb_dim)
    sum_cat_emb_dim = sum(col_to_emb_dim.values())
    self.linear1    = nn.Linear(sum_cat_emb_dim + num_var_number, linear_size)
    self.linear2    = nn.Linear(linear_size, class_number)
    self.dropout    = nn.Dropout(dropout_rate)
    
  def forward(self, num_variables, *cat_variables):
    cat_embeddings = self.cat_emb(cat_variables)
    cat_num_tensor = torch.cat([cat_embeddings, num_variables], dim = 1)
    cat_num_tensor = self.dropout(cat_num_tensor)
    out_linear1    = F.relu(self.dropout(self.linear1(cat_num_tensor)))
    out_linear2    = self.linear2(out_linear1)

    return F.log_softmax(out_linear2, dim=1)
