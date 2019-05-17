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
    'AGYSUB': 'Agency',
    'LOC': 'Location',
    'AGELVL': 'Age (bucket)',
    'EDLVL': 'Education level',
    'GSEGRD': 'General schedule & Equivalent grade',
    'LOSLVL': 'Length of service (bucket)',
    'OCC': 'Occupation',
    'PATCO': 'Occupation category',
    'PPGRD': 'Pay Plan & Grade',
    'STEMOCC': 'STEM Occupation',
    'SUPERVIS': 'Supervisory status',
    'TOA': 'Type of appointment',
    'WORKSCH': 'Work schedule',
    'WORKSTAT': 'Work status',
    'LOS': 'Average length of service',
    'SALBUCKET': 'Salary bucket'
}

def clean_dataframe(df):
    # Removing the nan values in columns by either adding a new category
    # or dropping the lines
    df                                   = df[~df.EDLVL.isnull()]
    df.loc[df.GSEGRD.isnull(), 'GSEGRD'] = 0
    df.loc[df.OCC.isnull(), 'OCC']       = 0
    df                                   = df[~df.SUPERVIS.isnull()]
    df                                   = df[~df.TOA.isnull()]
    df                                   = df[~df.SALARY.isnull()]
    df                                   = df[~df.LOS.isnull()]
    # Target generation, we partition the salary values in 10 equally sized
    # buckets.
    df['SALBUCKET']                      = pd.qcut(df.SALARY, np.arange(0, 1.1, .1))

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

    return df_cat, df_num, df_target, column_order

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

if __name__ == '__main__':
    data_path                 = Path('../../data/')
    model_folder              = Path('../../models/')
    model_path                = model_folder / 'edlvl_clf_salary_bucket.pt'
    emb_dim                   = 5
    df                        = pd.read_csv(data_path / 'interim' / 'fed_emp.csv')
    df                        = clean_dataframe(df)
    df                        = df.sample(10000, random_state = seed)
    df_data, df_target        = split_input_target(df)
    (
        df_cat,
        df_num,
        df_target,
        column_order
    ) = encode_and_normalize_columns(df_data, df_target)
    dataset                   = create_pytorch_dataset(
        df_cat,
        df_num,
        df_target,
        column_order
    )
    print(dataset[0])
