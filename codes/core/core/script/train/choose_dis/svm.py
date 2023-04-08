import itertools
import numpy as np
from copy import deepcopy

from core.predict.model_testor import ModelTestor
from core.reader.hpo_reader import HPOReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.predict.ml_model.SVMModel import generate_model, LSVMConfig
import core.predict.ml_model.SVMModel as lsvm
from core.utils.utils import get_logger, delete_logger
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_EMBEDDING, TRAIN_MODE
from core.utils.constant import PHELIST_REDUCE, PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP, VEC_COMBINE_MEAN, VEC_COMBINE_MAX, VEC_COMBINE_SUM
from core.utils.constant import CHOOSE_DIS_GEQ_HPO, CHOOSE_DIS_GEQ_IC, RESULT_PATH, TEST_DATA, RESULT_FIG_PATH
from core.helper.data.data_helper import DataHelper
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, get_embed_mat, multi_tune, get_slice_hpo_reader


def train_svm_model(d, save_model=False, model_name=None):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']

	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)

	if 'choose_dis_mode' in d:
		hpo_reader = get_slice_hpo_reader(
			d['choose_dis_mode'], min_hpo=d.get('min_hpo', None),
			min_IC=d.get('min_IC', None), slice_phe_list_mode=d.get('slice_phe_list_mode', None))
	else:
		hpo_reader = HPOReader()

	if vec_type == VEC_TYPE_EMBEDDING:
		embed_mat = get_embed_mat(d['embedInfo'][0], **d['embedInfo'][1])
		model = generate_model(vec_type, hpo_reader=hpo_reader, phe_list_mode=phe_list_mode, embed_mat=embed_mat, combine_modes=d['combine_modes'], mode=TRAIN_MODE, model_name=model_name)
		del d['embedInfo'], d['combine_modes']
	else:
		model = generate_model(vec_type, hpo_reader=hpo_reader, phe_list_mode=phe_list_mode, mode=TRAIN_MODE, model_name=model_name)
	model.train(LSVMConfig(d), save_model=save_model)
	return model


def get_grid():

	grid = {
		'C':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
		'vec_type':[VEC_TYPE_0_1],
		'choose_dis_mode':[CHOOSE_DIS_GEQ_IC],
		'min_IC':[0, 15, 30, 50, 70, 90, 110, 150, 200, 250, 300, 350, 400],
		'slice_phe_list_mode': [PHELIST_ANCESTOR]
	}
	return grid


def tune_svm_script():
	grid = get_grid()
	model_name = lsvm.LSVMModel.__name__
	hyp_save_folder = RESULT_PATH + '/ChooseDis/{}'.format(model_name)

	multi_tune(grid, train_svm_model, model_name, search_type='grid', cpu_use=12, test_cpu_use=12, hyp_save_folder=hyp_save_folder)


def cal_metric_and_draw(key='min_hpo'):
	grid = get_grid()
	model_name = lsvm.LSVMModel.__name__
	hyp_save_folder = RESULT_PATH + '/ChooseDis/{}'.format(model_name)
	hyper_helper = HyperTuneHelper(
		model_name,
		save_folder=hyp_save_folder
	)
	hyper_helper.draw_score_with_iteration()
	hyper_helper.draw_score_with_para('min_hpo')
	hyper_helper.draw_score_with_para('min_IC')

	data_names = ['MME', 'RAMEDIS', 'CJFH', 'PC']
	metric_names = ['TopAcc.10', 'TopAcc.1', 'RankMedian']
	model_names = []
	for v in grid[key]:
		model_names.append('{}-{}-{}'.format('svm', key, v))


	ModelTestor(TEST_DATA).draw_metric_bar(data_names, metric_names, model_names, fig_dir=RESULT_FIG_PATH + '/Barplot/chooseDis/svm')


if __name__ == '__main__':
	pass

	cal_metric_and_draw(key='min_IC')


