import numpy as np
from copy import deepcopy
from scipy.sparse import vstack

from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.semi import GMMModel, GMMConfig
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, TRAIN_MODE
from core.helper.data.data_helper import DataHelper
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, get_default_dtype


def train_gmm_model(d):
	print(d)
	d = deepcopy(d)

	vec_type = d['vec_type']
	phe_list_mode = get_default_phe_list_mode(vec_type)
	xdtype = get_default_dtype(vec_type)
	dh = DataHelper()
	u_X, uy_ = dh.getCombinedUTrainXY(d['u_data_names'], phe_list_mode, vec_type, xdtype=xdtype, min_hpo=d['min_hpo'], del_dup=d['del_dup'], x_sparse=True)
	X, y_ = dh.get_train_XY(phe_list_mode, vec_type, xdtype=xdtype, x_sparse=True, y_one_hot=False)
	X, y_ = vstack([X, u_X]), np.hstack([y_, uy_])
	del d['vec_type'], d['u_data_names'], d['min_hpo'], d['del_dup']

	gmm_config = GMMConfig(d)
	model = GMMModel(vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE)
	model.train_X(X.toarray(), y_, gmm_config, save_model=False)
	return model


def tune_gmm_script():
	grid = {
		'min_hpo': [3],
		'del_dup': [True, False],
		'u_data_names': [('U_DEC', )],
		'cov_type': ['diag', 'spherical', 'tied'],
		'vec_type': [VEC_TYPE_0_1, VEC_TYPE_TF, VEC_TYPE_TF_IDF],
	}
	hyper_helper = HyperTuneHelper(GMMModel.__name__, 'a')
	tune(grid, train_gmm_model, hyper_helper, 'grid')
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_para('del_dup')
	hyper_helper.draw_score_with_para('u_data_names')
	hyper_helper.draw_score_with_para('cov_type')
	hyper_helper.draw_score_with_para('vec_type')


if __name__ == '__main__':

	pass


