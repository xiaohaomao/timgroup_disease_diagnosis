

import numpy as np

from core.reader import HPOFilterDatasetReader
from core.predict.prob_model import TransferProbModel
from core.predict.prob_model import TransferProbNoisePunishModel
from core.utils.constant import SET_SIM_SYMMAX, SET_SIM_ASYMMAX_DQ, SET_SIM_ASYMMAX_QD, PHELIST_REDUCE, PHELIST_ANCESTOR, DIST_SHORTEST, DIST_MEAN_TURN
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.script.train.valid_tune.tune import tune, multi_tune
from core.utils.constant import TEST_DATA, VALIDATION_DATA, RESULT_PATH


def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])


def train_tf_prob_model(d, hpo_reader):
	return TransferProbModel(hpo_reader, **d)


def tune_prob_model():
	hpo_reader = get_hpo_reader()
	para_grid = {
		'default_prob': list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'alpha': list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)),
	}
	multi_tune(para_grid, train_tf_prob_model, TransferProbModel.__name__ + '-sum_miss-alpha_log', hpo_reader=hpo_reader, search_type='random', max_iter=200, cpu_use=20, test_cpu_use=20)


def train_tf_prob_noise_model(d, hpo_reader):
	return TransferProbNoisePunishModel(hpo_reader, **d)


def tune_prob_noise_model():
	hpo_reader = get_hpo_reader()
	para_grid = {
		'default_prob': list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
	}
	multi_tune(para_grid, train_tf_prob_noise_model, TransferProbNoisePunishModel.__name__, hpo_reader=hpo_reader, search_type='grid', cpu_use=15, test_cpu_use=15)



if __name__ == '__main__':
	tune_prob_model() # d02 rareDis2






