

import numpy as np

from core.reader import HPOFilterDatasetReader, HPOReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE
from core.predict.sim_model import RDDModel
from core.script.train.valid_tune.tune import tune

def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])


def train_rdd_model(d, hpo_reader):
	model = RDDModel(hpo_reader, **d)
	return model


def tune_rdd():
	hpo_reader = get_hpo_reader()
	grid = {
		'phe_list_mode': [PHELIST_ANCESTOR, PHELIST_REDUCE],
	}
	tune(grid, train_rdd_model, RDDModel.__name__, hpo_reader=hpo_reader, search_type='grid', cpu_use=8)


if __name__ == '__main__':
	tune_rdd()







