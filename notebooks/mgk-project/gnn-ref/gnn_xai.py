import os
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import sonnet as snt
import itertools
import collections
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_attribution import tasks
from graph_attribution import graphnet_models as gnn_models
from graph_attribution import graphnet_techniques as techniques
from graph_attribution import datasets, experiments, templates, hparams, training
from graph_attribution import graphs as graph_utils

datasets.DATA_DIR = os.path.join('../../../', 'data')


class InputArgs(Tap):
    task_name: str = 'logic7'
    seed: int = 0
    model: Literal['gcn', 'gat', 'mpnn', 'graphnet']
    # xai_technique: Literal['Random', 'GradInput', 'SmoothGrad(GradInput)', 'GradCAM-last', 'GradCAM-all', 'CAM']
    alpha: float = 0.01

    @property
    def dir(self):
        return f'{self.task_name}-{self.model}-{self.seed}'


args = InputArgs().parse_args()
if not os.path.exists(args.dir):
    os.mkdir(args.dir)
task_type = args.task_name
block_type = args.model

task_dir = datasets.get_task_dir(task_type)
exp, task, methods = experiments.get_experiment_setup(task_type, block_type)
task_act, task_loss = task.get_nn_activation_fn(), task.get_nn_loss_fn()
graph_utils.print_graphs_tuple(exp.x_train)

hp = hparams.get_hparams({'block_type': block_type, 'task_type': task_type})
print(f'\nhyperparameters: \n{hp}')
model = experiments.GNN(node_size=hp.node_size,
                        edge_size=hp.edge_size,
                        global_size=hp.global_size,
                        y_output_size=task.n_outputs,
                        block_type=gnn_models.BlockType(hp.block_type),
                        activation=task_act,
                        target_type=task.target_type,
                        n_layers=hp.n_layers)
model(exp.x_train)
gnn_models.print_model(model)
optimizer = snt.optimizers.Adam(hp.learning_rate)
opt_one_epoch = training.make_tf_opt_epoch_fn(exp.x_train, exp.y_train, hp.batch_size, model,
                                              optimizer, task_loss)
losses = collections.defaultdict(list)
start_time = time.time()
for _ in tqdm(range(hp.epochs)):
    train_loss = opt_one_epoch(exp.x_train, exp.y_train).numpy()
    losses['train'].append(train_loss)
    losses['test'].append(task_loss(exp.y_test, model(exp.x_test)).numpy())
    # pbar.set_postfix({key: values[-1] for key, values in losses.items()})

losses = {key: np.array(values) for key, values in losses.items()}

for key, values in losses.items():
    plt.plot(values, label=key)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.savefig(f'{args.dir}/training_loss.png')

results = []
for m, method in methods.items():
    result, pred_atts = experiments.generate_result_new(model, method, task, exp.x_test, exp.y_test, exp.att_test)
    datasets.save_graphtuples(f'{args.dir}/{args.task_name}-{args.model}-{m}-{args.seed}.npz', pred_atts)
    open(f'{args.dir}/{m}.log', 'w').write(json.dumps(result))
    results.append(experiments.generate_result(model, method, task, exp.x_test, exp.y_test, exp.att_test))
pd.DataFrame(results).to_csv(f'{args.dir}/results.log', index=False)
