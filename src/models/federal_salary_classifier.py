import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

seed = 142856
torch.manual_seed(seed)

class CategoricalEmbeddings(nn.Module):
    def __init__(self, col_order, col_encoders, col_to_emb_dim):
        super(CategoricalEmbeddings, self).__init__()
        self.col_order = col_order
        self.cat_embs  = nn.ModuleDict({
            col: nn.Embedding(len(col_encoders[col]), col_to_emb_dim[col])
            for col in col_order
        })

    def forward(self, cat_variables):
        embeddings = [self.cat_embs[col](cat_variables[col]) for col in self.col_order]

        return torch.cat(embeddings, dim = 1)

class NNClassifier(nn.Module):
    def __init__(self, col_order, col_encoders, col_to_emb_dim, class_number, num_var_number,
                 lin_size = 256, dropout_rate = 0.):
        super(NNClassifier, self).__init__()
        self.cat_emb    = CategoricalEmbeddings(col_order, col_encoders, col_to_emb_dim)
        sum_cat_emb_dim = sum(col_to_emb_dim.values())
        self.linear1    = nn.Linear(sum_cat_emb_dim + num_var_number, lin_size)
        self.linear2    = nn.Linear(lin_size, class_number)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, cat_variables, num_variables):
        cat_embeddings = self.cat_emb(cat_variables)
        cat_num_tensor = torch.cat([cat_embeddings, num_variables], dim = 1)
        cat_num_tensor = self.dropout(cat_num_tensor)
        out_linear1    = F.relu(self.dropout(self.linear1(cat_num_tensor)))
        out_linear2    = self.linear2(out_linear1)

        return out_linear2
