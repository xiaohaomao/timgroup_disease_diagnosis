"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np

from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE, VALIDATION_DATA, TEST_DATA
from core.predict.prob_model import BOQAModel
from core.script.train.valid_tune.tune import tune


def get_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=False)

def get_eval_datas():

	return [VALIDATION_DATA, TEST_DATA]


def train_boqa_model(d, hpo_reader):
	model = BOQAModel(hpo_reader, **d)
	return model


def tune_boqa(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	grid = {
		'dp': [None], # list(np.linspace(0.001, 0.009, 9)), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
		'use_freq': [True],
	}
	tune(grid, train_boqa_model, BOQAModel.__name__, hpo_reader=hpo_reader, search_type='grid', cpu_use=8, eval_datas=get_eval_datas())


if __name__ == '__main__':
	from core.script.train.valid_tune.tune import resort_history_for_model, combine_history_for_model



	for keep_dnames in [['OMIM', 'ORPHA', 'CCRD']]:
		tune_boqa(keep_dnames)





