import pandas as pd
from graph_attribution import tasks
from graph_attribution import datasets
import os
import numpy as np
from math import sqrt
from typing import Dict, List, Set, Tuple, Union
import torch
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import interp
from scipy.stats import pearsonr, spearmanr
CWD = os.path.dirname(os.path.abspath(__file__))


def get_data(task_name):
    task_type = tasks.Task(task_name)
    task = tasks.get_task(task_type)
    # fnames = datasets.get_default_experiment_filenames(task_type)
    task_path = os.path.join(CWD, '../..', 'data', task_name)

    train_index, test_index = datasets.load_train_test_indices(
        os.path.join(task_path,
                     '%s_traintest_indices.npz' % task_name)
    )
    att = datasets.load_graphstuples(
        os.path.join(task_path,
                     'true_raw_attribution_datadicts.npz')
    )
    att_test = [att[i] for i in test_index]
    y = datasets.load_npz(
        os.path.join(task_path,
                     'y_true.npz')
    )['y'][:, 0]
    df = pd.read_csv(
        os.path.join(task_path,
                     '%s_smiles.csv' % task_name)
    )
    df['y'] = y.astype(int)
    df_train = df[df.index.isin(train_index)].reset_index().drop(columns=['index'])
    df_test = df[df.index.isin(test_index)].reset_index().drop(columns=['index'])
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    return df_train, df_test, att_test, task
