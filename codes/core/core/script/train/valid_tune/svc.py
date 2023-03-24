

import itertools
import numpy as np
from copy import deepcopy

from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.ml_model import SVCModel, SVCConfig
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_EMBEDDING, TRAIN_MODE
from core.utils.constant import PHELIST_REDUCE, PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP, VEC_COMBINE_MEAN, VEC_COMBINE_MAX, VEC_COMBINE_SUM
from core.utils.constant import TEST_DATA, VALIDATION_DATA, RESULT_PATH
from core.helper.data.data_helper import DataHelper
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, get_embed_mat, multi_tune


def train_svc_model(d, save_model=False, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']

	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)

	if vec_type == VEC_TYPE_EMBEDDING:
		embed_mat = get_embed_mat(d['embedInfo'][0], **d['embedInfo'][1])
		model = SVCModel(vec_type=vec_type, phe_list_mode=phe_list_mode, embed_mat=embed_mat, combine_modes=d['combine_modes'], mode=TRAIN_MODE, model_name=model_name)
		del d['embedInfo'], d['combine_modes']
	else:
		model = SVCModel(vec_type=vec_type, phe_list_mode=phe_list_mode, mode=TRAIN_MODE, model_name=model_name)
	model.train(SVCConfig(d), save_model=save_model)
	return model


def get_grid():
	grid = {
		'C': list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10))
			+ list(np.linspace(0.01, 0.1, 10)) + list(np.linspace(0.1, 1.0, 10)),
		'vec_type':[VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF],
	}
	return grid


def tune_svc_script():
	grid = get_grid()
	multi_tune(grid, train_svc_model, SVCModel.__name__, search_type='random', max_iter=100, cpu_use=8, test_cpu_use=12)
	for eval_data in [TEST_DATA, VALIDATION_DATA]:
		hyp_save_name = SVCModel.__name__
		hyper_helper = HyperTuneHelper(hyp_save_name, save_folder=RESULT_PATH + '/hyper_tune/{}/{}'.format(hyp_save_name, eval_data))
		hyper_helper.draw_score_with_iteration()
		hyper_helper.draw_score_with_para('C')
		hyper_helper.draw_score_with_para('vec_type')


if __name__ == '__main__':


	import numpy as np
	from core.utils.constant import EMBEDDING_PATH, get_tune_data_names, get_tune_metric_names
	from core.predict.model_testor import ModelTestor

	# d03 rareDis5
	embed_mat = np.load(EMBEDDING_PATH + '/GDJointEncoder-loss1-loss2/GDJointEncoder-COMBIND_HPO_AVG-PHELIST_REDUCE-1.0-0.5-0.1-256-[]/embedding.npy')
	for C, gamma in itertools.product([0.01, 0.1, 1.0, 10.0, 100.0], ['auto', 'scale']):
		model = SVCModel(vec_type=VEC_TYPE_EMBEDDING, phe_list_mode=PHELIST_REDUCE, embed_mat=embed_mat,
			combine_modes=[VEC_COMBINE_MEAN], mode=TRAIN_MODE, model_name='svm-gdj')
		model.train(SVCConfig({'C': C, 'gamma': gamma}), save_model=False)
		mt = ModelTestor(TEST_DATA)
		mt.load_test_data()


