

import numpy as np

from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import PHE_SIM_MINIC, PHE_SIM_MICA
from core.predict.sim_model import GDDPFisherModel
from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.script.train.valid_tune.tune import tune, multi_tune
from core.utils.constant import TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, RESULT_PATH


def get_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=False)


def get_eval_datas():

	return [VALIDATION_TEST_DATA, VALIDATION_DATA, TEST_DATA]


# ================================================================
def train_fisher_model(d, hpo_reader):
	model = GDDPFisherModel(hpo_reader, **d)
	return model


def train_random_fisher_model(d, hpo_reader):
	model = OrderedMultiModel([
		(GDDPFisherModel, (hpo_reader,), d),
		(RandomModel, (hpo_reader,), {'seed': 777})], hpo_reader=hpo_reader)
	return model


def tune_fisher(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	para_grids = [

		{
			'phe_sim':[PHE_SIM_MICA],
			'gamma':list(np.linspace(0, 5, 51))
		}
	]
	for para_grid in para_grids:

		tune(para_grid, train_random_fisher_model, GDDPFisherModel.__name__ + '_mica_random',
			hpo_reader=hpo_reader, search_type='grid', cpu_use=8, eval_datas=get_eval_datas())


if __name__ == '__main__':
	from core.script.train.valid_tune.tune import resort_history_for_model, combine_history_for_model

	tune_fisher()








