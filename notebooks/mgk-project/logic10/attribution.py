import sys
sys.path.append('..')
import os
from utils import get_data
from tap import Tap
from mgktools.interpret.interpret import get_interpreted_mols
from graph_attribution.datasets import save_graphtuples, load_graphstuples
from sklearn.metrics import roc_auc_score
import numpy as np


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
# dataset
df_train, df_test, att_test, task = get_data(args.task_name)
print('there are %d molecules in the test set' % len(df_test))

# attribution accuracy estimation
pred_att_npz = '%s/mgk_attribution.npz' % dir_
if not os.path.exists(pred_att_npz):
    smiles_train = df_train['smiles'].tolist()
    targets_train = df_train['y'].tolist()
    smiles_to_be_interpret = df_test['smiles'].tolist()
    assert len(smiles_to_be_interpret) != 0
    mols = get_interpreted_mols(smiles_train=smiles_train,
                                targets_train=targets_train,
                                smiles_to_be_interpret=smiles_to_be_interpret,
                                mgk_hyperparameters_file='%s/hyperparameters_0.json' % dir_,
                                alpha=args.alpha,
                                batch_size=10)
    from graph_attribution.featurization import mol_to_graphs_tuple, MolTensorizer
    import numpy as np
    tensorizer = MolTensorizer(read_interpretation=True)
    pred_atts = [mol_to_graphs_tuple([mol], tensorizer) for mol in mols]
    save_graphtuples(pred_att_npz, pred_atts)
else:
    pred_atts = load_graphstuples(pred_att_npz)

for i in range(len(pred_atts)):
    assert np.sum(abs(pred_atts[i].senders - att_test[i].senders)) == 0
    assert np.sum(abs(pred_atts[i].receivers - att_test[i].receivers)) == 0
result = task.evaluate_attributions(att_test, pred_atts, reducer_fn=np.nanmean)
open('%s/att-auc' % dir_, 'w').write(str(result))
att_true = []
att_pred = []
for i, gt in enumerate(att_test):
    att_true += gt.nodes[:, -1].tolist()
    att_pred += pred_atts[i].nodes.tolist()
result = roc_auc_score(att_true, att_pred)
open('%s/full_att-auc' % dir_, 'w').write(str(result))
