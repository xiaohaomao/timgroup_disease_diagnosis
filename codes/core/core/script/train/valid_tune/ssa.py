
import numpy as np
import os
from copy import deepcopy

from core.predict.deep_model.StructuredSelfAttModel import SSAConfig, generate_model, StructuredSelfAttentionModel
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, TRAIN_MODE, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE
from core.utils.constant import TEST_DATA, VALIDATION_DATA, RESULT_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, ATT_DOT, ATT_ADD, ATT_MULTIPLY
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, train_best_model


def get_train_model(d, model_name=None):
	d = deepcopy(d)
	phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	c = SSAConfig(d)
	model = generate_model(phe_list_mode=phe_list_mode, model_name=model_name, mode=TRAIN_MODE)
	return model, c


def get_predict_model(d, model_name=None):
	phe_list_mode = d['phe_list_mode']
	return generate_model(phe_list_mode=phe_list_mode, model_name=model_name)


def train_model(d, model_name=None, save_model=False, **kwargs):

	model, c = get_train_model(d, model_name)
	model.train(c, save_model=save_model, from_last=False)
	model = get_predict_model(d, model_name)
	return model


def get_grid():
	grid = {
		'w_decay':[0.0],
		'phe_list_mode': [PHELIST_ANCESTOR],
		'r': [10, 30],
		'u': [64, 128, 256],
		'd_a': [128, 256],
		'p_coef': [0.0] + list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) +
		         list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'max_epoch_num': [400],
		'lr': [0.002, 0.001, 0.0005, 0.0001],
		'clip_grad': [5.0],

		'transform': [True, False],
		'head': [1, 2],
		'att_type': [ATT_DOT],
	}
	return grid


def tune_script():
	grid = get_grid()
	tune(grid, train_model, StructuredSelfAttentionModel.__name__, search_type='random', max_iter=200)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # d03 rareDis4
	tune_script()



