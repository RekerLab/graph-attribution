import sys

sys.path.append('..')
import os
from utils import get_data
from tap import Tap
from mgktools.data.data import Dataset
from graphdot.model.gaussian_process import GaussianProcessRegressor as GPR
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mgktools.kernels.utils import get_kernel_config


class InputArgs(Tap):
    task_name: str = 'logic7'
    seed: int = 0
    n = 'best'
    alpha: float = 0.01
    dir: str = 'hyperopt_loocv'


args = InputArgs().parse_args()
if args.n == 'best':
    dir_ = '%s/%d' % (args.dir, args.seed)
else:
    dir_ = '%s/%d/%s' % (args.dir, args.seed, args.n)

if os.path.exists('%s/pred.csv' % dir_) and os.path.exists('%s/pred-acc' % dir_):
    exit(0)

# dataset
df_train, df_test, att_test, task = get_data(args.task_name)
if os.path.exists('train.pkl'):
    train = Dataset.load(path='', filename='train.pkl')
else:
    train = Dataset.from_df(
        df_train,
        pure_columns=['smiles'],
        target_columns=['y'],
        n_jobs=8
    )
    train.save(path='', filename='train.pkl')

if os.path.exists('test.pkl'):
    test = Dataset.load(path='', filename='test.pkl')
else:
    test = Dataset.from_df(
        df_test,
        pure_columns=['smiles'],
        target_columns=['y'],
        n_jobs=8
    )
    test.save(path='', filename='test.pkl')

train.graph_kernel_type = 'graph'
test.graph_kernel_type = 'graph'
# prediction accuracy estimation
kernel_config = get_kernel_config(
    train,
    graph_kernel_type='graph',
    # arguments for marginalized graph kernel
    mgk_hyperparameters_files=['%s/hyperparameters_0.json' % dir_],
)
gpr = GPR(kernel=kernel_config.kernel, alpha=args.alpha).fit(train.X, train.y)
y_pred = gpr.predict(test.X)
df = pd.DataFrame({'y_pred': y_pred, 'y_truth': test.y})
df.to_csv('%s/pred.csv' % dir_, index=False)
open('%s/pred-acc' % dir_, 'w').write('%.5f' % accuracy_score(test.y, np.rint(y_pred)))
