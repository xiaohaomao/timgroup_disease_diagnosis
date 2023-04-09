import os
import numpy as np
from copy import deepcopy
from core.reader import HPOReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.prob_model import MNBConfig, CNBConfig
from core.predict.prob_model import MNBModel, CNBModel, TreeMNBModel, HPOProbMNBModel
from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, TRAIN_MODE, RESULT_PATH, TEST_DATA, VALIDATION_DATA
from core.utils.constant import PHELIST_ANCESTOR_DUP, PHELIST_ANCESTOR, PHELIST_REDUCE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE
from core.utils.constant import TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA
from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, multi_tune


def get_hpo_reader(keep_dnames=None, rm_no_use_hpo=False):

	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=rm_no_use_hpo)


def get_eval_datas():
	return [VALIDATION_DATA, TEST_DATA, VALIDATION_TEST_DATA]


# ================================================================
def train_mnb_model(d, hpo_reader, save_model=False, model_name=None):
	d = deepcopy(d)
	test01 = d.get('test01', False); del d['test01']

	mnb_config = MNBConfig(d)
	model = MNBModel(hpo_reader=hpo_reader, mode=TRAIN_MODE, model_name=model_name)
	model.train(mnb_config, save_model=save_model)
	if test01:
		model.phe_list_mode = PHELIST_ANCESTOR
		model.raw_X_to_X_func = model.raw_X_to_01_X
	return model


def train_random_mnb_model(d, hpo_reader, save_model=False, model_name=None):
	model = train_mnb_model(d, hpo_reader, save_model=save_model, model_name=model_name)
	return OrderedMultiModel(hpo_reader=hpo_reader, model_list=[model, RandomModel(hpo_reader=hpo_reader, seed=777)], model_name=model_name)


def tune_mnb_script():
	hpo_reader = get_hpo_reader()
	grid = {
		'alpha': list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
				+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'test01': [True, False]
	}

	tune(grid, train_random_mnb_model, MNBModel.__name__+'Random', hpo_reader=hpo_reader, search_type='grid',
		max_iter=100, eval_datas=get_eval_datas())


def train_best_mnb():
	d = {
		'alpha': 0.02,
		'test01': True
	}
	train_mnb_model(d, get_hpo_reader(), save_model=True, model_name='MNB')


# ================================================================
def train_tree_mnb_model(d, save_model=False):
	d = deepcopy(d)
	test01 = d.get('test01', False); del d['test01']

	model = TreeMNBModel(mode=TRAIN_MODE, p=d['p']); del d['p']
	mnb_config = MNBConfig(d)
	model.train(mnb_config, None, save_model=save_model)
	if test01:
		model.phe_list_mode = PHELIST_ANCESTOR
		model.raw_X_to_X_func = model.raw_X_to_01_X
	return model


def tune_tree_mnb_script():
	grid = {
		'alpha': list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
				+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'p': list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
				+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'test01':[True, False],
	}

	tune(grid, train_tree_mnb_model, TreeMNBModel.__name__, search_type='random', max_iter=100)
	hyper_helper = HyperTuneHelper(TreeMNBModel.__name__)
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_iteration(
		figpath=hyper_helper.HISTORY_FOLDER+'/ScoreWithIterTest01True.png',
		filter=lambda h: h[0].get('test01', False) == True)
	hyper_helper.draw_score_with_iteration(
		figpath=hyper_helper.HISTORY_FOLDER + '/ScoreWithIterTest01False.png',
		filter=lambda h:h[0].get('test01', False) == False)
	hyper_helper.draw_score_with_para('alpha')
	hyper_helper.draw_score_with_para('p')
	hyper_helper.draw_score_with_para('test01')


def train_best_tree_mnb():
	d = {
		'alpha': 0.08105527638190954,
		'p': 0.0037688442211055275,
		'test01': True
	}
	train_tree_mnb_model(d, save_model=True)


# ================================================================
def train_hpo_prob_mnb_model(d, hpo_reader):
	return HPOProbMNBModel(hpo_reader, **d)


def train_random_hpo_prob_mnb_model(d, hpo_reader):
	model = OrderedMultiModel([
		(HPOProbMNBModel, (hpo_reader,), d),
		(RandomModel, (hpo_reader,), {'seed':777})], hpo_reader=hpo_reader)
	return model


def tune_hpo_prob_mnb_model(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames)
	para_grids = [{
		'phe_list_mode': [PHELIST_REDUCE], # PHELIST_ANCESTOR
		'p1': list(np.linspace(0.001, 0.009, 9)) + list(np.linspace(0.01, 0.09, 9)) + list(np.linspace(0.1, 1.0, 19)),
		'p2': [None],
		'child_to_parent_prob': ['sum', 'max', 'ind'],

	}]
	for grid in para_grids:
		# multi_tune(grid, train_hpo_prob_mnb_model, 'HPOProbMNB', search_type='random', max_iter=100, cpu_use=10, test_cpu_use=10)
		multi_tune(grid, train_random_hpo_prob_mnb_model, 'HPOProbMNB-Random',
			hpo_reader=hpo_reader, search_type='random', max_iter=111, cpu_use=12, test_cpu_use=12, eval_datas=get_eval_datas())


# ================================================================
def train_cnb_model(d, hpo_reader, save_model=False, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	use_rd_mix_code = d['use_rd_mix_code']; del d['use_rd_mix_code']
	test01 = d.get('test01', False); del d['test01']
	cnb_config = CNBConfig(d)
	phe_list_mode = get_default_phe_list_mode(vec_type)

	model = CNBModel(hpo_reader=hpo_reader, vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE,
		model_name=model_name, use_rd_mix_code=use_rd_mix_code)
	model.train(cnb_config, save_model=save_model)
	if test01:
		model.phe_list_mode = PHELIST_ANCESTOR
		model.raw_X_to_X_func = model.raw_X_to_01_X
	return model


def tune_cnb_script(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames, rm_no_use_hpo=False)
	grid = {
		'alpha': list(np.linspace(0.1, 0.9, 9)) + list(np.linspace(1.0, 9.0, 9))
		         + list(np.linspace(10.0, 90.0, 9)) + list(np.linspace(100.0, 1000.0, 19)),

		'test01': [False],  # True is the same as False
		'use_rd_mix_code': [False],
		'vec_type':[VEC_TYPE_0_1], # VEC_TYPE_0_1, VEC_TYPE_TF
	}

	tune(grid, train_cnb_model, CNBModel.__name__, hpo_reader=hpo_reader,
		search_type='grid', max_iter=100, eval_datas=get_eval_datas())

def train_best_cnb(keep_dnames=None):
	source_to_d = {
		'PHENOMIZERDIS': { # same with INTEGRATE_CCRD_OMIM_ORPHA
			"alpha":350.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_CCRD_OMIM_ORPHA': { # CCRD_OMIM_ORPHA: 150.0 200.0, 100.0, # 80.0
			"alpha":350.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_OMIM_ORPHA':{
			"alpha":200.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_CCRD_OMIM':{
			"alpha":100.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_CCRD_ORPHA':{
			"alpha":250.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_OMIM':{
			"alpha":90.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_ORPHA':{
			"alpha":150.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'INTEGRATE_CCRD':{
			"alpha":5.0,
			"test01":False,
			"use_rd_mix_code":False,
			"vec_type":"VEC_TYPE_0_1"
		},
		'CCRD_OMIM_ORPHA': {},
		'OMIM_ORPHA':{},
		'CCRD_OMIM':{},
		'CCRD_ORPHA':{},
		'OMIM':{},
		'ORPHA':{},
		'CCRD':{},
	}
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames, rm_no_use_hpo=False)
	d = source_to_d[hpo_reader.name]
	train_cnb_model(d, hpo_reader=hpo_reader, save_model=True, model_name='CNB')


def train_best_cnb_web():
	d = {
		"alpha":150.0,
		"test01":False,
		"use_rd_mix_code":False,
		"vec_type":"VEC_TYPE_0_1"
	}
	hpo_reader = get_hpo_reader(rm_no_use_hpo=False)
	train_cnb_model(d, hpo_reader=hpo_reader, save_model=True, model_name='CNB-Web')


if __name__ == '__main__':

	pass
