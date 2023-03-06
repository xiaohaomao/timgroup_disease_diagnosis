"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.predict.model_testor import ModelTestor
from core.predict.RFModel import generate_model, RFConfig
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, DATA_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, LOG_PATH, PREDICT_MODE, TRAIN_MODE, VEC_TYPE_0_1_DIM_REDUCT
from core.utils.constant import PHELIST_ANCESTOR, RESULT_PATH
from core.utils.utils import get_logger, getStandardRawXAndY
import os
from tqdm import tqdm
from multiprocessing import Pool
import itertools


def process(para):
	rf_config, model_name = para

	hpo_reader = HPOReader()
	raw_X, y_ = getStandardRawXAndY(hpo_reader, PHELIST_ANCESTOR)
	sw = None

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, rf_config, None, save_model=False)

	return model


def train_script():
	tree_num_list = [100, 1000, 5000, 10000]
	criterion_list = ['gini']
	max_leaf_node_list = [2**5-1, 2**6-1, 2**7-1]

	paras = []
	for tn, cri, mln in itertools.product(tree_num_list, criterion_list, max_leaf_node_list):
		rf_config = RFConfig()
		rf_config.n_estimators = tn
		rf_config.criterion = cri
		rf_config.max_leaf_nodes = mln
		model_name = 'RFModel_01_Ances_tn{}_cri{}_mln{}'.format(tn, cri, mln)
		paras.append((rf_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)



if __name__ == '__main__':

	tree_num_list = [100, 1000, 5000]
	criterion_list = ['gini']
	max_leaf_node_list = [2**5-1, 2**6-1, 2**7-1]
	ll = ['RFModel_01_Ances_tn{}_cri{}_mln{}'.format(tn, cri, mln) for tn, cri, mln in itertools.product(tree_num_list, criterion_list, max_leaf_node_list)]








