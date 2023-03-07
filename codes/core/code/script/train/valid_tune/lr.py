"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import numpy as np
from copy import deepcopy

from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.predict.ml_model import LogisticModel, LogisticConfig
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, VEC_TYPE_EMBEDDING, PHELIST_REDUCE, PREDICT_MODE
from core.utils.constant import PHELIST_ANCESTOR_DUP, PHELIST_ANCESTOR, LOG_PATH, VEC_COMBINE_MEAN, VEC_COMBINE_SUM, VEC_COMBINE_MAX
from core.utils.constant import TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, RESULT_PATH, HYPER_TUNE_SUCC_Z_SCORE
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, multi_tune, get_embed_mat, train_best_model
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper


def get_hpo_reader():

	return HPOIntegratedDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'], rm_no_use_hpo=True)


def get_eval_datas():
	return [VALIDATION_DATA, TEST_DATA, VALIDATION_TEST_DATA]



def get_model(d, hpo_reader, model_name, mode):

	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)

	if vec_type == VEC_TYPE_EMBEDDING:
		embed_mat = get_embed_mat(d['embedInfo'][0], **d['embedInfo'][1])
		model = LogisticModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, embed_mat=embed_mat,
			combine_modes=d['combine_modes'], model_name=model_name, mode=mode, use_rd_mix_code=use_rd_mix_code)
		del d['embedInfo'], d['combine_modes']
	else:
		model = LogisticModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode,
			model_name=model_name, mode=mode, use_rd_mix_code=use_rd_mix_code)
	return model


def get_predict_model(d, hpo_reader, model_name):
	return get_model(d, hpo_reader, model_name, PREDICT_MODE)


def train_lr_model(d, hpo_reader, save_model=False, model_name=None, **kwargs):
	print(d)
	d = deepcopy(d)
	model = get_model(d, hpo_reader, model_name, TRAIN_MODE)
	lr_config = LogisticConfig(d)
	model.train(lr_config, save_model=save_model)
	return model


def train_best_lr_model(d, hpo_reader, model_name, repeat=5, use_query_many=True, test_cpu_use=12):
	return train_best_model(
		d, train_lr_model, lambda d, model_name: get_model(d, hpo_reader, model_name, PREDICT_MODE), model_name,
		repeat=repeat, use_query_many=use_query_many, test_cpu_use=test_cpu_use
	)


def get_grid():
	grid = {
		'C': list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
		    + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)) + list(np.linspace(1.0, 10.0, 10)),
		'use_rd_mix_code': [False],
		'fit_intercept': [True],
		'vec_type': [VEC_TYPE_0_1],
	}



	return grid


def tune_lr_script():
	grid = get_grid()
	hpo_reader = get_hpo_reader()
	multi_tune(grid, train_lr_model, LogisticModel.__name__ + '-01-DelHPO',
		hpo_reader=hpo_reader, search_type='grid', max_iter=100, cpu_use=10, test_cpu_use=10, eval_datas=get_eval_datas())


def train_best_sim_vec_lr():

	d = {
		"C": 0.00001,
		"fit_intercept": True,
		"vec_type": "VEC_TYPE_0_1",
		"use_rd_mix_code": False,
	}
	train_lr_model(d, get_hpo_reader(), save_model=True, model_name='LR-fuck', repeat=1)


def train_best_embed_vec_lr1():
	# [0.160, 0.4, 0.089]
	d = {
		"C": 0.005,
		"fit_intercept": False,
		"vec_type": "VEC_TYPE_EMBEDDING",
		"embedInfo": [
			"glove", {
				"embed_size": 512,
				"x_max": 50,
				"phe_list_mode": "PHELIST_ANCESTOR_DUP"
			}
		],
		"phe_list_mode": "PHELIST_REDUCE",
		"combine_modes": [
			"VEC_COMBINE_MAX",
			"VEC_COMBINE_SUM"
		]
	}
	train_lr_model(d, get_hpo_reader(), save_model=True, model_name='LR-Glove')


if __name__ == '__main__':

	tune_lr_script()

