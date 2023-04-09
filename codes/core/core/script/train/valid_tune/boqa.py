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
		'dp': [None],
		'use_freq': [True],
	}
	tune(grid, train_boqa_model, BOQAModel.__name__, hpo_reader=hpo_reader, search_type='grid', cpu_use=8, eval_datas=get_eval_datas())


if __name__ == '__main__':

	pass



