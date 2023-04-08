import numpy as np
import os
from copy import deepcopy
import scipy, random

from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.reader import HPOFilterDatasetReader, HPOReader, HPOIntegratedDatasetReader
from core.reader import HPOReader, RDFilterReader
from core.predict.ml_model import LRNeuronConfig, LRNeuronModel
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE
from core.utils.constant import TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, RESULT_PATH, OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_RMS
from core.utils.constant import HYPER_TUNE_AVE_SCORE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE, SEED
from core.utils.utils import random_vec
from core.script.train.valid_tune.tune import tune, multi_tune, get_default_phe_list_mode, train_best_model


def get_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=True)


def get_mt_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=True)


def get_eval_datas():
	return [VALIDATION_DATA, TEST_DATA]


def get_train_model(d, hpo_reader, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	phe_list_mode = get_default_phe_list_mode(vec_type)
	c = LRNeuronConfig(d)
	c.n_features = hpo_reader.get_hpo_num()
	c.class_num = RDFilterReader(keep_source_codes=hpo_reader.get_dis_list()).get_rd_num() if use_rd_mix_code else hpo_reader.get_dis_num()
	model = LRNeuronModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode,
		model_name=model_name, mode=TRAIN_MODE, use_rd_mix_code=use_rd_mix_code)
	return model, c


def get_predict_model(d, hpo_reader, model_name=None, init_para=True):
	vec_type = d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']
	phe_list_mode = get_default_phe_list_mode(vec_type)
	return LRNeuronModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode,
		model_name=model_name, init_para=init_para, use_rd_mix_code=use_rd_mix_code)


def train_lr_model(d, hpo_reader, model_name=None, save_model=False, **kwargs):

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	model_name = 'NN-Mixup-Tune-10.0'

	model, c = get_train_model(d, hpo_reader, model_name)
	model.train(c, save_model=save_model, from_last=False)

	model.restore()
	return OrderedMultiModel(
			hpo_reader=hpo_reader, model_list=[model, RandomModel(hpo_reader=hpo_reader, seed=777)],
			keep_raw_score=False, model_name=model_name+'-Random')


def train_best_lr_model(d, hpo_reader, model_name, repeat=10, use_query_many=True, cpu_use=2, test_cpu_use=12, use_pool=True):
	return train_best_model(d, train_lr_model, get_predict_model, model_name, hpo_reader=hpo_reader, repeat=repeat,
		use_query_many=use_query_many, cpu_use=cpu_use, test_cpu_use=test_cpu_use, use_pool=use_pool,
		mt_hpo_reader=get_mt_hpo_reader())


def get_grid():


	grid = {
		'w_decay': [0.0, 1e-9, 1e-8, 1e-7, 1e-6],
		'lr':[0.0006, 0.0008, 0.001, 0.0015, 0.002],
		'vec_type':[VEC_TYPE_0_1],
		'use_rd_mix_code':[False],
		'multi_label':[False],
		'mixup':[True],
		'seed':[2211],
		'hyper_score_type': [HYPER_TUNE_RANK_SCORE],

		'mix_alpha': [10.0],
		'mt_test_freq': [100],
		'early_stop_patience': [10],

	}


	return grid


def tune_lr_script(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	grid = get_grid()
	tune(grid, train_lr_model, LRNeuronModel.__name__+'-Mixup-Tune-10.0', hpo_reader=hpo_reader,
		search_type='random', max_iter=25, cpu_use=1, use_pool=False,
		mt_hpo_reader=get_mt_hpo_reader(keep_dnames=keep_dnames), eval_datas=get_eval_datas())



def train_best_sim_lr1_concat():
	d = {
		"w_decay": 1e-07,
		"keep_prob": 1.0,
		"lr": 0.0015,
		"vec_type": "VEC_TYPE_0_1",
		"hyper_score_type": "HYPER_TUNE_Z_SCORE",
		"seed": 777,
		"use_rd_mix_code": False,
		"multi_label": False,
		"mt_test_freq": 100,
		"early_stop_patience": 10
	}
	train_best_lr_model(d, get_hpo_reader(), model_name='NN-1', repeat=1, cpu_use=1)

def train_best_mixup_lr1_concat():
	d = { #
		"w_decay": 1e-07,
		"lr": 0.0008,
		"vec_type": "VEC_TYPE_0_1",
		"use_rd_mix_code": False,
		"multi_label": False,
		"mixup": True,
		"seed": 2211,
		"hyper_score_type": "HYPER_TUNE_Z_SCORE",
		"mix_alpha": 2.0,
		"mt_test_freq": 100,
		"early_stop_patience": 10
	}
	train_best_lr_model(d, get_hpo_reader(), model_name='NN-Mixup-1', repeat=1, cpu_use=1, use_pool=False)

def train_best_pertur_lr1_concat():
	d = {
		"w_decay": 1e-07,
		"lr": 0.0008,
		"perturbation": True,
		"pertur_weight": [
			0.5,
			0.05097704026491775,
			0.016367995817052144,
			0.212176847223811,
			0.22047811669421907
		],
		"mixup": False,
		"seed": 2211,
		"multi_label": False,
		"use_rd_mix_code": False,
		"vec_type": "VEC_TYPE_0_1",
		"mt_test_freq": 100,
		"early_stop_patience": 10
	}
	train_best_lr_model(d, get_hpo_reader(), model_name='NN-Pert-1', repeat=1, cpu_use=1, use_pool=False)

def train_best_mixup_lr_web_concat():
	d = {
		"w_decay":1e-07,
		"lr":0.0006,
		"vec_type":"VEC_TYPE_0_1",
		"use_rd_mix_code":False,
		"multi_label":False,
		"mixup":True,
		"seed":2211,
		"hyper_score_type":"HYPER_TUNE_Z_SCORE",
		"mix_alpha":0.1,
		"mt_test_freq":100,
		"early_stop_patience":10
	}
	train_best_lr_model(d, get_hpo_reader(), model_name='NN-Mixup-Web', repeat=1, cpu_use=1, use_pool=False)


# knowledge integrate ==================================================================================================
def train_best_sim_lr1(keep_dnames=None):
	source_to_d = {
		'INTEGRATE_CCRD_OMIM_ORPHA':{
			"w_decay":1e-07,
			"keep_prob":1.0,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"seed":2928,
			"use_rd_mix_code":False,
			"multi_label":False,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_OMIM_ORPHA':{},
		'INTEGRATE_CCRD_OMIM':{},
		'INTEGRATE_CCRD_ORPHA':{},
		'INTEGRATE_OMIM':{},
		'INTEGRATE_ORPHA':{},
		'INTEGRATE_CCRD':{},
		'CCRD_OMIM_ORPHA': {

		}
	}
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	d = source_to_d[hpo_reader.name]
	train_best_lr_model(d, hpo_reader, model_name='NN-1', repeat=1, cpu_use=1)


def train_best_mixup_lr1(keep_dnames=None):
	source_to_d = {
		'PHENOMIZERDIS':{  # same with INTEGRATE_CCRD_OMIM_ORPHA
			"w_decay":1e-07,
			"lr":0.001,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":2.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_CCRD_OMIM_ORPHA': {
			"w_decay":1e-07,
			"lr":0.001,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":2.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_OMIM_ORPHA': {

			"w_decay":1e-07,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2212,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":0.1,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_CCRD_OMIM': {
			"w_decay":1e-07,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":1.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_CCRD_ORPHA': {
			"w_decay":1e-07,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":2.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_OMIM': {
			"w_decay":1e-07,
			"lr":0.0006,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":10.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_ORPHA': {
			"w_decay":1e-07,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":2.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_CCRD': {
			"w_decay":1e-06,
			"lr":0.0015,
			"vec_type":"VEC_TYPE_0_1",
			"use_rd_mix_code":False,
			"multi_label":False,
			"mixup":True,
			"seed":2211,
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mix_alpha":1.0,
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'CCRD_OMIM_ORPHA':{},
		'OMIM_ORPHA':{},
		'CCRD_OMIM':{},
		'CCRD_ORPHA':{},
		'OMIM':{},
		'ORPHA':{},
		'CCRD':{},
	}

	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	d = source_to_d[hpo_reader.name]
	train_best_lr_model(d, hpo_reader, model_name='NN-Mixup-1', repeat=1, cpu_use=1, use_pool=False)


def train_best_mixup_lr_web():
	d = {
		"w_decay":1e-08,
		"lr":0.0008,
		"vec_type":"VEC_TYPE_0_1",
		"use_rd_mix_code":False,
		"multi_label":False,
		"mixup":True,
		"seed":2211,
		"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
		"mix_alpha":10.0,
		"mt_test_freq":100,
		"early_stop_patience":10
	}
	hpo_reader = get_hpo_reader()
	train_best_lr_model(d, hpo_reader, model_name='NN-Mixup-Web', repeat=1, cpu_use=1, use_pool=False)


def train_best_pertur_lr1(keep_dnames=None):
	source_to_d = {
		'INTEGRATE_CCRD_OMIM_ORPHA':{
			"w_decay":0.0,
			"lr":0.0015,
			"perturbation":True,
			"pertur_weight":[
				0.5,
				0.1448696071195164,
				0.0797173967363264,
				0.25649793321548375,
				0.018915062928673468
			],
			"mixup":False,
			"seed":2211,
			"multi_label":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1",
			"hyper_score_type":"HYPER_TUNE_RANK_SCORE",
			"mt_test_freq":100,
			"early_stop_patience":10
		},
		'INTEGRATE_OMIM_ORPHA':{},
		'INTEGRATE_CCRD_OMIM':{},
		'INTEGRATE_CCRD_ORPHA':{},
		'INTEGRATE_OMIM':{},
		'INTEGRATE_ORPHA':{},
		'INTEGRATE_CCRD':{},
		'CCRD_OMIM_ORPHA':{

		}
	}
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	d = source_to_d[hpo_reader.name]
	train_best_lr_model(d, hpo_reader, model_name='NN-Pert-1', repeat=1, cpu_use=1, use_pool=False)


if __name__ == '__main__':
	from core.script.train.valid_tune.tune import resort_history_for_model, combine_history_for_model



	tune_lr_script()



