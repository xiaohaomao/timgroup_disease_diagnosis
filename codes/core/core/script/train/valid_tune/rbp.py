import numpy as np

from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.predict.sim_model import RBPModel, RBPDominantRandomModel
from core.script.train.valid_tune.tune import tune
from core.utils.constant import VALIDATION_TEST_DATA, VALIDATION_DATA, TEST_DATA


def get_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=False)


def get_eval_datas():
	return [VALIDATION_DATA, TEST_DATA, VALIDATION_TEST_DATA]



def train_rbp_model(d, hpo_reader):
	model = RBPModel(hpo_reader, **d)
	return model


def train_rbp_dominate_random_model(d, hpo_reader):
	model = OrderedMultiModel([
		(RBPModel, (hpo_reader,), d),
		(RandomModel, (hpo_reader,), {'seed':777})], hpo_reader=hpo_reader)
	return model


def tune_rbp(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	grid = {
		'alpha': list(np.linspace(0.1, 1.0, 10)) + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.001, 0.01, 10)),
	}

	tune(grid, train_rbp_dominate_random_model, 'RBPModel-Random',
		hpo_reader=hpo_reader, search_type='grid', cpu_use=8, eval_datas=get_eval_datas())


if __name__ == '__main__':
	tune_rbp()




