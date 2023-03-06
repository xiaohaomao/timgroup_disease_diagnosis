"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.predict.sim_model.DistanceModel import DistanceModel, generate_model
from core.utils.constant import SET_SIM_SYMMAX, SET_SIM_ASYMMAX_DQ, SET_SIM_ASYMMAX_QD, PHELIST_REDUCE, PHELIST_ANCESTOR, DIST_SHORTEST, DIST_MEAN_TURN
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.script.train.valid_tune.tune import tune, multi_tune

def train_model(d):
	d['init_cpu_use'] = 6
	return generate_model(**d)


def tune_model():
	para_grid = {
		'set_sim_method': [SET_SIM_SYMMAX, SET_SIM_ASYMMAX_DQ, SET_SIM_ASYMMAX_QD],
		'phe_list_mode': [PHELIST_REDUCE, PHELIST_ANCESTOR],
		'dist_type': [DIST_SHORTEST, DIST_MEAN_TURN]
	}
	tune(para_grid, train_model, DistanceModel.__name__, search_type='grid', cpu_use=6)
	hyper_helper = HyperTuneHelper(DistanceModel.__name__, 'a')
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_para('set_sim_method')
	hyper_helper.draw_score_with_para('phe_list_mode')
	hyper_helper.draw_score_with_para('dist_type')


if __name__ == '__main__':
	tune_model()






