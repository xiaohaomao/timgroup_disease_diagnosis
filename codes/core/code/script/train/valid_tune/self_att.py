"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np
import os
from copy import deepcopy

from core.predict.deep_model.SelfAttentionModel import AttConfig, generate_model, SelfAttentionModel
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE
from core.utils.constant import TEST_DATA, VALIDATION_DATA, RESULT_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, ATT_DOT, ATT_ADD, ATT_MULTIPLY
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, train_best_model


def get_train_model(d, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	c = AttConfig(d)
	model = generate_model(vec_type=vec_type, phe_list_mode=phe_list_mode, model_name=model_name, mode=TRAIN_MODE)
	return model, c


def get_predict_model(d, model_name=None):
	vec_type = d['vec_type']
	phe_list_mode = d['phe_list_mode']
	return generate_model(vec_type=vec_type, phe_list_mode=phe_list_mode, model_name=model_name)


def train_model(d, model_name=None, save_model=False, **kwargs):

	model, c = get_train_model(d, model_name)
	model.train(c, save_model=save_model, from_last=False)
	model = get_predict_model(d, model_name)
	return model


def get_grid():
	grid = {
		'w_decay':[0.0, 1e-9, 1e-8, 1e-7],
		'perturbation':[True],
		'pertur_weight':[[0.5, 0.05, 0.15, 0.2, 0.1]],
		'vec_type':[VEC_TYPE_0_1],
		'phe_list_mode': [PHELIST_REDUCE],
		'att_type': [ATT_DOT],
		'head_num': [1, 2, 4],
		'block_num': [1, 2],
		'HSize': [128],
		'ff_units': [256],
		'f_layers': [[]],
		'g_layers': [[]],
	}
	return grid


def tune_script():
	grid = get_grid()
	tune(grid, train_model, SelfAttentionModel.__name__, search_type='grid', max_iter=100)
	for eval_data in [TEST_DATA, VALIDATION_DATA]:
		hyp_save_name = SelfAttentionModel.__name__
		hyper_helper = HyperTuneHelper(hyp_save_name, save_folder=RESULT_PATH + '/hyper_tune/{}/{}'.format(hyp_save_name, eval_data))
		for scoreSortType in [HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE]:
			hyper_helper.draw_score_with_iteration(sort_type=scoreSortType)
			hyper_helper.draw_score_with_para('phe_list_mode', sort_type=scoreSortType)
			hyper_helper.draw_score_with_para('head_num', sort_type=scoreSortType)
			hyper_helper.draw_score_with_para('block_num', sort_type=scoreSortType)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # d03 rareDis4
	tune_script()









