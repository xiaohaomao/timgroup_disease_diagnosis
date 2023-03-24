
from core.reader import HPOFilterDatasetReader
from core.predict.sim_model import MICAModel
from core.utils.constant import SET_SIM_SYMMAX, SET_SIM_ASYMMAX_DQ, SET_SIM_ASYMMAX_QD, VALIDATION_DATA
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.script.train.valid_tune.tune import tune


def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'], rm_no_use_hpo=False)


def train_MICA_model(d, hpo_reader):
	return MICAModel(hpo_reader=hpo_reader, **d)


def tune_MICA():
	hpo_reader = get_hpo_reader()
	para_grid = {
		'set_sim_method': [SET_SIM_SYMMAX, SET_SIM_ASYMMAX_DQ, SET_SIM_ASYMMAX_QD],
	}
	tune(para_grid, train_MICA_model, MICAModel.__name__, hpo_reader=hpo_reader, search_type='grid')


if __name__ == '__main__':
	tune_MICA()  # d03 rareDis






