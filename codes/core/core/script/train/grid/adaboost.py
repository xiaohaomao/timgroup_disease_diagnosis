from core.predict.model_testor import ModelTestor
from core.predict.ml_model.AdaBoostModel import generate_model, AdaConfig
from core.utils.constant import VEC_TYPE_0_1, TRAIN_MODE
from core.utils.constant import PHELIST_ANCESTOR
from core.helper.data.data_helper import DataHelper

import os
from tqdm import tqdm
from multiprocessing import Pool
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def process(para):
	ada_config, model_name = para

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, ada_config, None, save_model=False)

	return model


def train_script():
	cls_num_list = [50, 100, 200, 300, 500, 1000]
	lr_list = [1.0, 0.8, 0.6]
	algorithms = ['SAMME.R', 'SAMME']

	paras = []
	for clsNum, lr, alg in itertools.product(cls_num_list, lr_list, algorithms):
		ada_config = AdaConfig()
		ada_config.n_estimators = clsNum
		ada_config.lr = lr
		ada_config.algorithm = alg

		model_name = 'AdaBoostModel_01_Ances_clsNum{}_lr{}_alg{}'.format(clsNum, lr, alg)
		paras.append((ada_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	with Pool(20) as pool:
		for model in tqdm(pool.imap_unordered(process, paras), total=len(paras), leave=False):
			for data_name in mt.data_names:
				mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def train_script2():
	lr_max_iter_list = [100]
	cls_num_list = [10, 50, 100]  #
	lr_list = [1.0]
	algorithms = ['SAMME.R']

	paras = []
	for lrMaxIter, clsNum, lr, alg in itertools.product(lr_max_iter_list, cls_num_list, lr_list, algorithms):
		ada_config = AdaConfig()
		ada_config.base_estimator = LogisticRegression(max_iter=lrMaxIter)
		ada_config.n_estimators = clsNum
		ada_config.lr = lr
		ada_config.algorithm = alg

		model_name = 'AdaBoostModel_01_Ances_LR_max_iter{}_clsNum{}_lr{}_alg{}'.format(lrMaxIter, clsNum, lr, alg)
		paras.append((ada_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def train_script3():
	cls_num_list = [100, 500, 1000, 2000] #
	lr_list = [1.0]
	algorithms = ['SAMME.R']
	max_leaf_node_list = [2**5-1, 2**6-1, 2**7-1]

	paras = []
	for clsNum, lr, alg, mln in itertools.product(cls_num_list, lr_list, algorithms, max_leaf_node_list):
		ada_config = AdaConfig()
		ada_config.base_estimator = DecisionTreeClassifier(max_leaf_nodes=mln)
		ada_config.n_estimators = clsNum
		ada_config.lr = lr
		ada_config.algorithm = alg

		model_name = 'AdaBoostModel_01_Ances_clsNum{}_lr{}_alg{}_mln{}'.format(clsNum, lr, alg, mln)
		paras.append((ada_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True, chunk_size=200)


if __name__ == '__main__':
	cls_num_list = [100, 500, 1000, 2000] #
	lr_list = [1.0]
	algorithms = ['SAMME.R']
	max_leaf_node_list = [2**5-1, 2**6-1, 2**7-1]









