import sys
sys.path.append('..')
import os
from tap import Tap
import pandas as pd
from mgktools.interpret.interpret import get_interpreted_mols
from graph_attribution.datasets import save_graphtuples, load_graphstuples
from rdkit import Chem
from mgktools.data.data import Dataset
from graph_attribution.featurization import mol_to_graphs_tuple, MolTensorizer
from graph_attribution.tasks import BinaryClassificationTaskType
import numpy as np
from ..utils import calc_metric, norm


class InputArgs(Tap):
    data_path: str
    seed: int = 0
    n = 'best'
    alpha: float = 0.01
    dir: str = 'hyperopt_loocv'


args = InputArgs().parse_args()
df = pd.read_csv(args.data_path)
df_train = df[df.splits == 'train']
df_test = df[df.splits == 'test']
if args.n == 'best':
    dir_ = '%s/%d' % (args.dir, args.seed)
else:
    dir_ = '%s/%d/%s' % (args.dir, args.seed, args.n)
# dataset
print('there are %d molecules in the test set' % len(df_test))

if not os.path.exists('attribution_gt.npz'):
    tensorizer = MolTensorizer(read_interpretation=True)
    atts = np.load('attributions.npz', allow_pickle=True)['attributions']
    mols = []
    for i, smiles in enumerate(df_test.smiles):
        mol = Chem.MolFromSmiles(smiles)
        idx = df_test.iloc[i].name
        assert atts[idx]['SMILES'] == smiles
        assert len(mol.GetAtoms()) == len(atts[idx]['node_atts'])
        for j, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atomNote', str(atts[idx]['node_atts'][j]))
        mols.append(mol)
    att_test = [mol_to_graphs_tuple([mol], tensorizer) for mol in mols]
    save_graphtuples('attribution_gt.npz', att_test)
else:
    att_test = load_graphstuples('attribution_gt.npz')

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

    tensorizer = MolTensorizer(read_interpretation=True)
    pred_atts = [mol_to_graphs_tuple([mol], tensorizer) for mol in mols]
    save_graphtuples(pred_att_npz, pred_atts)
else:
    pred_atts = load_graphstuples(pred_att_npz)

y_truth = []
y_pred = []
for i in range(len(pred_atts)):
    assert np.sum(abs(pred_atts[i].senders - att_test[i].senders)) == 0
    assert np.sum(abs(pred_atts[i].receivers - att_test[i].receivers)) == 0
    y_truth += att_test[i].nodes.tolist()
    y_pred += pred_atts[i].nodes.tolist()
    # y_pred += norm(pred_atts[i].nodes).tolist()
# y_pred = norm(y_pred)
result = calc_metric(y_truth, y_pred, 'auc')
print('AUC: %f' % calc_metric(y_truth, y_pred, 'auc'))
print('ACC: %f' % calc_metric(y_truth, y_pred, 'acc'))
open('%s/att-auc' % dir_, 'w').write(str(result))
