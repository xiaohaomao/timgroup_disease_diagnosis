import itertools
import numpy as np
import os
from copy import deepcopy

from core.reader import HPOFilterDatasetReader, RDFilterReader
from core.predict.deep_model.hmcn_model import HMCNModel, HMCNConfig
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, PREDICT_MODE, RELU, SIGMOID, TANH
from core.utils.constant import HYPER_TUNE_Z_SCORE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_AVE_SCORE
from core.utils.utils import random_vec
from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode


def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])


def train_hmcn_model(d, hpo_reader, model_name=None):
	print(d)
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	unit_acts = d['unit_acts']; del d['unit_acts']

	phe_list_mode = get_default_phe_list_mode(vec_type)
	c = HMCNConfig(d)
	c.unit_act_lists = [unit_acts] * len(c.level_orders_list)
	c.n_features = hpo_reader.get_hpo_num()

	model_name = 'HMCN-Tune'
	model = HMCNModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE,
		model_name=model_name, use_rd_mix_code=use_rd_mix_code)
	model.train(c, save_model=False, from_last=False)
	model.restore(); model.mode = PREDICT_MODE
	return model


def get_grid():
	grid = {
		'w_decay': [0.0],
		'keep_prob': [0.3, 0.2],
		'lr': [0.001, 0.005, 0.01],
		'unit_acts': [ [(unit, act)] * layers for layers, unit, act in
			itertools.product([0, 1], [4096], [TANH])],
		'vec_type': [VEC_TYPE_0_1],
		'use_rd_mix_code': [False],
		'dis_restrict': [False],
		'res_connect': [True],
		'beta': [0.0],
		'seed': [2211],
		'mt_test_freq': [200],
		'early_stop_patience': [10],
		'ances_dp': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
		'hyper_score_type': [HYPER_TUNE_AVE_SCORE],
		'level_orders_list': [

			[
				[DISORDER_GROUP_LEVEL], [DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL],
				[DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
			],
			[
				[DISORDER_GROUP_LEVEL, DISORDER_LEVEL], [DISORDER_SUBTYPE_LEVEL],
				[DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
			],
		]
	}
	return grid


def tune_hmcn_script():
	hpo_reader = get_hpo_reader()
	grid = get_grid()
	tune(grid, train_hmcn_model, HMCNModel.__name__+'-Tune', hpo_reader=hpo_reader, search_type='random', max_iter=108)



if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	tune_hmcn_script()

