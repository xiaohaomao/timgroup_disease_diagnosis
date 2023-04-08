import numpy as np
import os
from copy import deepcopy

from core.predict.semi import SemiLRModel, SemiLRNeuronConfig
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE
from core.utils.utils import random_vec
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode


def train_semi_lr_model(d, model_name=None):
	print(d)
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	phe_list_mode = get_default_phe_list_mode(vec_type)
	c = SemiLRNeuronConfig(d)
	model = SemiLRModel(vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE, model_name=model_name)
	model.train(c, save_model=False, from_last=False)

	return model


def get_grid():

	grid = {
		'w_decay':[0.0, 1e-7, 1e-6, 1e-5, 1e-4],
		'perturbation': [False],

		'mixup': [False],

		'vec_type':[VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF],
		'u_lambda': list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10))
					+ list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)),
		'u_data_names': [('U_DEC',), ('U_CLINVAR_SCV',), ('U_DEC', 'U_CLINVAR_SCV')],
		'min_hpo':[3, 4, 5],
		'del_dup':[True],
	}
	return grid


def tune_semi_lr_script():
	grid = get_grid()
	hyper_helper = HyperTuneHelper(SemiLRModel.__name__, 'a')
	tune(grid, train_semi_lr_model, hyper_helper, search_type='random', max_iter=200)
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_para('u_lambda')
	hyper_helper.draw_score_with_para('w_decay')
	hyper_helper.draw_score_with_para('u_data_names')
	hyper_helper.draw_score_with_para('vec_type')


def train_best_sim_semi_lr():
	# [0.17730914157246444, 0.018010829296095755, -261.3333333333333]
	d = {
		"w_decay": 1e-07,
		"perturbation": False,
		"mixup": False,
		"vec_type": "VEC_TYPE_TF",
		"u_lambda": 3.0000000000000004e-05,
		"u_data_names": [
			"U_DEC"
		],
		"min_hpo": 4,
		"del_dup": True
	}
	train_semi_lr_model(d, model_name='SemiLR')


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	train_best_sim_semi_lr()



