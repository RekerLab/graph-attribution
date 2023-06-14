import sys
sys.path.append('..')
import os
from utils import get_data
from tap import Tap
from mgktools.data.data import Dataset
from mgktools.hyperparameters import additive_pnorm
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters.hyperopt import bayesian_optimization
from mgktools.data.split import dataset_split


class InputArgs(Tap):
    task_name: str = 'logic7'
    alpha: float = 0.01
    loss: str = 'loocv'
    num_iters: int = 30
    num_splits: int = 10
    seed: int = 0


args = InputArgs().parse_args()
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

# hyperparameter optimization


kernel_config = get_kernel_config(
    train,
    graph_kernel_type='graph',
    # arguments for marginalized graph kernel
    mgk_hyperparameters_files=[additive_pnorm],
)

dir_ = 'hyperopt_%s_%d' % (args.loss, args.num_splits)
if not os.path.exists(dir_):
    os.mkdir(dir_)
dir_ = 'hyperopt_%s_%d/%d' % (args.loss, args.num_splits, args.seed)
if not os.path.exists(dir_):
    os.mkdir(dir_)
train.graph_kernel_type = 'graph'
test.graph_kernel_type = 'graph'

if args.loss == 'loocv':
    best_hyperdict, results, hyperdicts = \
        bayesian_optimization(save_dir=dir_,
                              datasets=dataset_split(train, split_type='random', sizes=[1 / args.num_splits] * args.num_splits),
                              kernel_config=kernel_config,
                              model_type='gpr',
                              task_type='binary',
                              metric='roc-auc',
                              split_type='loocv',
                              num_iters=args.num_iters,
                              alpha=args.alpha,
                              alpha_bounds=None,
                              seed=args.seed
                              )
elif args.loss == 'log_likelihood':
    best_hyperdict, results, hyperdicts = \
        bayesian_optimization(save_dir=dir_,
                              datasets=dataset_split(train, split_type='random', sizes=[1 / args.num_splits] * args.num_splits),
                              kernel_config=kernel_config,
                              model_type='gpr',
                              task_type='regression',
                              metric='log_likelihood',
                              num_iters=args.num_iters,
                              alpha=args.alpha,
                              alpha_bounds=None,
                              seed=args.seed
                              )
else:
    raise ValueError()

open('%s/loss' % (dir_), 'w').write(str(max(results)))
for i, hyperdict in enumerate(hyperdicts):
    if not os.path.exists('%s/%d' % (dir_, i)):
        os.mkdir('%s/%d' % (dir_, i))
    kernel_config.update_from_space(hyperdict)
    kernel_config.save_hyperparameters('%s/%d' % (dir_, i))
    open('%s/%d/loss' % (dir_, i), 'w').write(str(results[i]))
