import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import numpy as np
from copy import deepcopy

from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.ml_model import LSVMModel, LSVMConfig
import core.predict.ml_model as lsvm
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_EMBEDDING, TRAIN_MODE
from core.utils.constant import PHELIST_REDUCE, PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP, VEC_COMBINE_MEAN, VEC_COMBINE_MAX, VEC_COMBINE_SUM
from core.utils.constant import TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, RESULT_PATH
from core.helper.data.data_helper import DataHelper
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, get_embed_mat, multi_tune


def get_hpo_reader():
	return HPOIntegratedDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'], rm_no_use_hpo=True)


def get_eval_datas():
	return [VALIDATION_DATA, TEST_DATA, VALIDATION_TEST_DATA]


def train_svm_model(d, hpo_reader, save_model=False, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)

	if vec_type == VEC_TYPE_EMBEDDING:
		embed_mat = get_embed_mat(d['embedInfo'][0], **d['embedInfo'][1])
		model = LSVMModel(
			hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, embed_mat=embed_mat,
			combine_modes=d['combine_modes'], mode=TRAIN_MODE, model_name=model_name, use_rd_mix_code=use_rd_mix_code)
		del d['embedInfo'], d['combine_modes']
	else:
		model = LSVMModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE,
			model_name=model_name, use_rd_mix_code=use_rd_mix_code)
	model.train(LSVMConfig(d), save_model=save_model)
	return model


def get_grid():
	grid = {
		'C': list(np.linspace(1e-7, 1e-6, 10)) + list(np.linspace(1e-6, 1e-5, 10))
			+ list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
			+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'use_rd_mix_code': [False],
		'vec_type':[VEC_TYPE_0_1],
	}

	return grid


def tune_svm_script():
	hpo_reader = get_hpo_reader()
	grid = get_grid()
	multi_tune(grid, train_svm_model, lsvm.LSVMModel.__name__ + '-01-DelHPO',
		hpo_reader=hpo_reader, search_type='grid', max_iter=100, cpu_use=10, test_cpu_use=10, eval_datas=get_eval_datas())



def train_best_svm_model():
	train_svm_model(HyperTuneHelper(lsvm.LSVMModel.__name__).get_best_para(), True)


def train_best_sim_vec_svm():

	d = {
		"C": 1e-06, # 0.002
		"use_rd_mix_code": False,
		"vec_type": "VEC_TYPE_0_1",
	}
	train_svm_model(d, get_hpo_reader(), save_model=True, model_name='SVM')


def train_best_embed_vec_svm1():
	# [0.195, 0.358, 0.079]
	d = {
		"C": 0.0002,
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
	train_svm_model(d, save_model=True, model_name='LSVM-Glove')


if __name__ == '__main__':

	tune_svm_script()



