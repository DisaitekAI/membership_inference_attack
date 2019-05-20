from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from federal_salary_classifier import CategoricalEmbeddings, NNClassifier

seed = 142856
torch.manual_seed(seed)

col_desc = {
    'AGYSUB'    : 'Agency',
    'LOC'       : 'Location',
    'AGELVL'    : 'Age (bucket)',
    'EDLVL'     : 'Education level',
    'GSEGRD'    : 'General schedule & Equivalent grade',
    'LOSLVL'    : 'Length of service (bucket)',
    'OCC'       : 'Occupation',
    'PATCO'     : 'Occupation category',
    'PPGRD'     : 'Pay Plan & Grade',
    'STEMOCC'   : 'STEM Occupation',
    'SUPERVIS'  : 'Supervisory status',
    'TOA'       : 'Type of appointment',
    'WORKSCH'   : 'Work schedule',
    'WORKSTAT'  : 'Work status',
    'LOS'       : 'Average length of service',
    'SALBUCKET' : 'Salary bucket'
}

def clean_dataframe(df, n_salary_slice = 10):
    # Removing the nan values in columns by either adding a new category
    # or dropping the lines
    df                                   = df[~df.EDLVL.isnull()]
    df.loc[df.GSEGRD.isnull(), 'GSEGRD'] = 0
    df.loc[df.OCC.isnull(), 'OCC']       = 0
    df                                   = df[~df.SUPERVIS.isnull()]
    df                                   = df[~df.TOA.isnull()]
    df                                   = df[~df.SALARY.isnull()]
    df                                   = df[~df.LOS.isnull()]
    # Target generation, we partition the salary values in
    # `n_salary_slice` equally sized buckets.
    slice_size                           = 1 / n_salary_slice
    df['SALBUCKET']                      = pd.qcut(
        df.SALARY,
        np.arange(
            0,
            1 + slice_size,
            slice_size
        )
    )
    print(df.SALBUCKET.unique())

    return df

def split_input_target(df):
    df_data    = df.drop(['SALBUCKET', 'SALARY', 'SALLVL'], axis = 1)
    df_target  = df['SALBUCKET']

    return df_data, df_target

def encode_and_normalize_columns(df_data, df_target):
    numerical_columns = ['LOS']
    # Numerical column normalization
    df_num            = df_data[numerical_columns]
    num_val_mean      = df_num.mean(axis = 0)
    num_val_std       = df_num.std(axis = 0)
    df_num            = (df_num - num_val_mean) / num_val_std

    # Categorical columns encoding
    df_cat            = df_data.drop(numerical_columns, axis = 1)
    columns_encoders = {
        col : {
            val : i
            for i, val in enumerate(df[col].unique())
        }
        for col in df_cat.columns
    }
    target_encoder = {
        val : i
        for i, val in enumerate(sorted(df_target.unique()))
    }
    column_order = sorted(columns_encoders.keys())

    for col in df_cat.columns:
        df_cat[col] = df_cat[col].apply(lambda x: columns_encoders[col][x])
    df_target = df_target.map(lambda x: target_encoder[x])

    return (df_cat, df_num, df_target, column_order, columns_encoders,
            target_encoder)

def create_pytorch_dataset(df_cat, df_num, df_target, column_order):
    dataset = TensorDataset(
        *[
            torch.tensor(df_cat[col].values)
            for col in column_order
        ], # categorical variables in the correct order
        torch.tensor(df_num.values, dtype = torch.float32), # numerical variables
        torch.tensor(df_target.values, dtype = torch.int64) # target variables
    )

    return dataset

def split_dataset(dataset, valid_prop = 0.2):
    dataset_size                 = len(dataset)
    valid_size                   = round(valid_prop * dataset_size)
    lengths                      = [dataset_size - valid_size, valid_size]
    train_dataset, valid_dataset = random_split(dataset, lengths)

    return train_dataset, valid_dataset

def create_model(column_order, column_encoders, target_encoder,
                 lin_size = 256, dropout_rate = 0., emb_dim = 5):
    model = NNClassifier(
        column_order,
        columns_encoders,
        {
            col : emb_dim
            for col in columns_encoders
        },
        len(target_encoder),
        num_var_number = df_num.shape[1],
        lin_size = lin_size,
        dropout_rate = dropout_rate
    )
    print(model)

    return model

def train_model(clf, epochs, train_loader, valid_loader, optimizer, device,
                column_order, model_path):
    for epoch in range(epochs):
        correct = 0
        total   = 0
        for i, (*cat_var_list, num_var, y) in enumerate(train_loader):
            optimizer.zero_grad()
            cat_var_list   = [t.to(device) for t in cat_var_list]
            num_var        = num_var.to(device)
            y              = y.to(device)
            cat_variables  = dict(zip(column_order, cat_var_list))
            res            = clf(cat_variables, num_var)
            loss           = criterion(res, y)
            correct       += (res.argmax(dim = 1) == y).detach().sum().item()
            total         += y.shape[0]
            loss.backward()
            optimizer.step()

        clf.eval()
        valid_correct = 0
        valid_total   = 0
        with torch.no_grad():
            for *cat_var_list, num_var, y in valid_loader:
                cat_var_list   = [t.to(device) for t in cat_var_list]
                num_var        = num_var.to(device)
                y              = y.to(device)
                cat_variables  = dict(zip(column_order, cat_var_list))
                res            = clf(cat_variables, num_var)
                valid_correct += (res.argmax(dim = 1) == y).detach().sum().item()
                valid_total   += y.shape[0]
        print(f'[{epoch}:{i}] [T] {100. * correct / total:5.2f}%, '
              f'[V] {100. * valid_correct / valid_total:5.2f}% '
              f'{loss.item():5.2f}')
        clf.train()

    torch.save(clf.state_dict(), model_path)

if __name__ == '__main__':
    dataset_trunc      = 10000
    data_path          = Path('../../data/')
    model_folder       = Path('../../models/')
    model_path         = model_folder / 'edlvl_clf_salary_bucket.pt'
    df                 = pd.read_csv(data_path / 'interim' / 'fed_emp.csv')
    df                 = clean_dataframe(df)
    df                 = df.sample(dataset_trunc, random_state = seed)
    df_data, df_target = split_input_target(df)
    (
        df_cat,
        df_num,
        df_target,
        column_order,
        columns_encoders,
        target_encoder
    ) = encode_and_normalize_columns(df_data, df_target)
    dataset            = create_pytorch_dataset(
        df_cat,
        df_num,
        df_target,
        column_order
    )
    (
        train_dataset,
        valid_dataset
    ) = split_dataset(dataset)
    train        = True
    epochs       = 100
    batch_size   = 2048
    criterion    = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    device       = torch.device('cuda')
    clf          = create_model(column_order, columns_encoders, target_encoder)
    clf          = clf.to(device)
    optimizer    = optim.Adam(clf.parameters())
    if train:
        train_model(
            clf,
            epochs,
            train_loader,
            valid_loader,
            optimizer,
            device,
            column_order,
            model_path
        )
    clf.load_state_dict(torch.load(model_path))
    clf.eval()
