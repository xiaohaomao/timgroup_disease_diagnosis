

import numpy as np

from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.prob_model import BayesNetModel
from core.script.train.valid_tune.tune import tune
from core.utils.constant import TRAIN_MODE, TEST_DATA, VALIDATION_DATA, RESULT_PATH

def train_bayes_net_model(d, save_model=False, model_name=None):
	print(d)
	model = BayesNetModel(mode=TRAIN_MODE, model_name=model_name, **d)
	model.train(save_model=save_model)
	return model


def tune_bayes_net_script():
	grid = {
		'alpha': list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
				+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'cond_type': ['max', 'ind']
	}
	tune(grid, train_bayes_net_model, BayesNetModel.__name__, search_type='grid', cpu_use=8)
	for eval_data in [TEST_DATA, VALIDATION_DATA]:
		hyp_save_name = BayesNetModel.__name__
		hyper_helper = HyperTuneHelper(hyp_save_name, save_folder=RESULT_PATH + '/hyper_tune/{}/{}'.format(hyp_save_name, eval_data))
		hyper_helper.draw_score_with_iteration()
		hyper_helper.draw_score_with_para('alpha')


def train_best_bayes_net0():
	d = {
		'alpha':0.8,
		'cond_type':'ind'
	}
	return train_bayes_net_model(d, save_model=True, model_name='BayesNet_ind_0.8')


def train_best_bayes_net1():
	d = {
		'alpha':0.4,
		'cond_type':'max'
	}
	return train_bayes_net_model(d, save_model=True, model_name='BayesNet_max_0.4')


def train_best_bayes_net2():
	d = {
		'alpha':0.4,
		'cond_type':'ind'
	}
	return train_bayes_net_model(d, save_model=True, model_name='BayesNet_ind_0.4')


if __name__ == '__main__':
	tune_bayes_net_script()    # d01 rareDis3


