"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.reader.hpo_reader import HPOReader
import core.predict.sim_model.mica_model as mica
from core.predict.model_testor import ModelTestor
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
import core.predict.ml_model.SVMModel as lsvm
from core.utils.constant import CHOOSE_DIS_GEQ_HPO, CHOOSE_DIS_GEQ_IC, RESULT_PATH, TEST_DATA, PHELIST_ANCESTOR
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_EMBEDDING, TRAIN_MODE
from core.utils.constant import RESULT_FIG_PATH
from core.utils.utils import get_logger, delete_logger
from core.script.train.valid_tune.tune import tune, get_default_phe_list_mode, get_embed_mat, multi_tune, get_slice_hpo_reader
from core.helper.hyper.para_grid_searcher import ParaGridSearcher

def train_MICA_model(d, model_name=None):
	if 'choose_dis_mode' in d:
		hpo_reader = get_slice_hpo_reader(
			d['choose_dis_mode'], min_hpo=d.get('min_hpo', None),
			min_IC=d.get('min_IC', None), slice_phe_list_mode=d.get('slice_phe_list_mode', None))
	else:
		hpo_reader = HPOReader()
	return mica.generate_model(hpo_reader=hpo_reader, model_name=model_name)


def get_grid():

	grid = {
		'choose_dis_mode':[CHOOSE_DIS_GEQ_IC],
		'min_IC':[0, 15, 30, 50, 70, 90, 110, 150, 200, 250, 300, 350, 400],
		'slice_phe_list_mode': [PHELIST_ANCESTOR]
	}
	return grid


def cal_metric_and_draw(key='min_hpo'):
	grid = get_grid()
	data_names = ['MME', 'RAMEDIS', 'CJFH', 'PC']
	metric_names = ['TopAcc.10', 'TopAcc.1', 'RankMedian']
	model_names = []
	for d in ParaGridSearcher(grid).iterator():
		v = d[key]
		model_names.append('{}-{}-{}'.format('MICA', key, v))

	ModelTestor(TEST_DATA).draw_metric_bar(data_names, metric_names, model_names, fig_dir=RESULT_FIG_PATH + '/Barplot/chooseDis/mica')


if __name__ == '__main__':
	pass

	cal_metric_and_draw(key='min_IC')







