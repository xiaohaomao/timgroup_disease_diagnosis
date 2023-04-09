import numpy as np
from copy import deepcopy
from scipy.sparse import vstack
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.semi import SemiMixMNConfig, SemiMixMNModel
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, TRAIN_MODE
from core.helper.data.data_helper import DataHelper
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, multi_tune


def train_mix_mn_model(d, save_model=False, model_name=None):
	print(d)
	d = deepcopy(d)
	vec_typeTrain, vec_type_test = d['vec_typeTrainTest']
	phe_listModeTrain, phe_list_mode_test = get_default_phe_list_mode(vec_typeTrain), get_default_phe_list_mode(vec_type_test)
	xdtype = np.int32
	dh = DataHelper()
	u_X, uy_ = dh.getCombinedUTrainXY(d['u_data_names'], phe_listModeTrain, vec_typeTrain, xdtype=xdtype, min_hpo=d['min_hpo'], del_dup=d['del_dup'], x_sparse=True)
	X, y_ = dh.get_train_XY(phe_listModeTrain, vec_typeTrain, xdtype=xdtype, x_sparse=True, y_one_hot=False)
	X, y_ = vstack([X, u_X]), np.hstack([y_, uy_])
	model = SemiMixMNModel(vec_type=vec_type_test, phe_list_mode=phe_list_mode_test, predict_pi=d['predict_pi'], mode=TRAIN_MODE, model_name=model_name)

	del d['vec_typeTrainTest'], d['u_data_names'], d['min_hpo'], d['del_dup']
	mn_config = SemiMixMNConfig(d)
	if mn_config.pi_init == 'random' or mn_config.UInit == 'random':
		mn_config.n_init = 3
	model.train_X(X, y_, mn_config, save_model=save_model, log_step=2)
	return model


def tune_mix_mn_script():
	grid = {
		'pi_init':['same', 'random'],
		'UInit':['TF', 'random', 'same'],
		'beta': list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)) + list(np.linspace(1.0, 10.0, 10)),
		'UAlpha': list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'predict_pi': ['same', 'tune'],

		'min_hpo': [3, 4, 5],
		'del_dup': [True],
		'u_data_names': [('U_DEC',), ('U_CLINVAR_SCV',), ('U_DEC', 'U_CLINVAR_SCV')],   # ('U_DEC', 'U_CLINVAR_SCV')
		'vec_typeTrainTest': [(VEC_TYPE_0_1, VEC_TYPE_0_1), (VEC_TYPE_TF, VEC_TYPE_TF), (VEC_TYPE_TF, VEC_TYPE_0_1)],
	}
	# tune(grid, train_mix_mn_model, hyper_helper, search_type='random', max_iter=200)
	multi_tune(grid, train_mix_mn_model, SemiMixMNModel.__name__, search_type='random', max_iter=300, cpu_use=6, test_cpu_use=6)
	hyper_helper = HyperTuneHelper(SemiMixMNModel.__name__, 'a')
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_para('min_hpo')
	hyper_helper.draw_score_with_para('del_dup')
	hyper_helper.draw_score_with_para('u_data_names')
	hyper_helper.draw_score_with_para('UAlpha')
	hyper_helper.draw_score_with_para('beta')


def valid(model):
	from core.predict.model_testor import ModelTestor
	from core.utils.constant import VALIDATION_DATA
	from core.script.train.valid_tune.tune import get_valid_score_dict
	mt = ModelTestor(VALIDATION_DATA)
	mt.load_test_data()
	return get_valid_score_dict(model, mt, use_query_many=True, cpu_use=12)


def train_best_mix_mn1():
	d = {
		"pi_init": "same",
		"UInit": "random",
		"beta": 0.08,
		"UAlpha": 0.05000000000000001,
		"predict_pi": "same",
		"min_hpo": 5,
		"del_dup": True,
		"u_data_names": [
			"U_DEC",
			"U_CLINVAR_SCV"
		],
		"vec_typeTrainTest": [
			"VEC_TYPE_TF",
			"VEC_TYPE_0_1"
		]
	}
	model = train_mix_mn_model(d, save_model=True, model_name='MixMN1')
	print(valid(model))


def train_best_mix_mn2():
	d = {
		"pi_init": "random",
		"UInit": "same",
		"beta": 0.6,
		"UAlpha": 0.05000000000000001,
		"predict_pi": "same",
		"min_hpo": 4,
		"del_dup": True,
		"u_data_names": [
			"U_DEC"
		],
		"vec_typeTrainTest": [
			"VEC_TYPE_TF",
			"VEC_TYPE_0_1"
		]
	}
	model = train_mix_mn_model(d, save_model=True, model_name='MixMN2')
	print(valid(model))


if __name__ == '__main__':

	pass