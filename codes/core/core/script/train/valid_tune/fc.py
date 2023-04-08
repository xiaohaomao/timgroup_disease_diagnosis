import itertools
import numpy as np
import os
from copy import deepcopy

from core.reader import HPOFilterDatasetReader, RDFilterReader
from core.predict.deep_model import FCConfig, FCModel
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, PREDICT_MODE, RELU, SIGMOID, TANH
from core.utils.utils import random_vec
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode


def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'], rm_no_use_hpo=True)


def train_fc_model(d, hpo_reader, model_name=None):
	print(d)
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	phe_list_mode = get_default_phe_list_mode(vec_type)
	c = FCConfig(d)
	c.n_features = hpo_reader.get_hpo_num()
	c.class_num = RDFilterReader(keep_source_codes=hpo_reader.get_dis_list()).get_rd_num() if use_rd_mix_code else hpo_reader.get_dis_num()

	model_name = 'FC-Tune-DelHPO'
	model = FCModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE,
		model_name=model_name, use_rd_mix_code=use_rd_mix_code)
	model.train(c, save_model=False, from_last=False)
	model.restore(); model.mode = PREDICT_MODE

	return model


def get_grid():
	grid = {
		'w_decay':[0.0],
		'keep_prob':[0.4, 0.6, 0.8],
		'lr':[0.0001],
		'unit_acts':[ [(unit, act)] * layers for layers, unit, act in
			itertools.product([1, 2], [8192], [RELU, TANH])],
		'vec_type':[VEC_TYPE_0_1],
		'use_rd_mix_code': [False],
		'multi_label':[False],

	}
	return grid


def tune_fc_script():
	hpo_reader = get_hpo_reader()
	grid = get_grid()
	tune(grid, train_fc_model, FCModel.__name__+'Tune-DelHPO', hpo_reader=hpo_reader, search_type='random', max_iter=12)



if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	tune_fc_script()

