"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, DATA_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, LOG_PATH, PREDICT_MODE, TRAIN_MODE
from core.utils.constant import PHELIST_ANCESTOR
from core.utils.utils import get_logger
import os
from core.predict.DecisionTreeModel import generate_model, DecisionTreeConfig

def train_script1():
	from core.utils.utils import read_train, read_train_from_files
	from sklearn.model_selection import train_test_split

	vec_type = VEC_TYPE_0_1
	embed_path = MODEL_PATH+'/DeepwalkModel/numwalks100/EMBEDDING'
	model = generate_model(vec_type, phe_list_mode=PHELIST_ANCESTOR, embed_path=embed_path)

	TRAIN_FOLDER = DATA_PATH + '/preprocess/AnnoDataSet10'
	# raw_X, y_ = read_train(DATA_PATH + '/preprocess/AnnoDataSet/true.txt')
	raw_X, y_, sw = read_train_from_files([
		TRAIN_FOLDER + '/true.txt',
		TRAIN_FOLDER + '/reduce.txt',
		TRAIN_FOLDER + '/rise-lower.txt',
		TRAIN_FOLDER + '/noise.txt',
	], file_weights=[1, 10, 1, 1], fix=False)


def process(paras):
	dt_config, model_name = paras

	hpo_reader = HPOReader()
	# One-shot =============================================
	dis_num = hpo_reader.get_dis_num()
	dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR)
	raw_X = [dis_int_to_hpo_int[i] for i in range(dis_num)]
	y_ = list(range(dis_num))
	sw = None
	# One-shot =============================================

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	logger = get_logger(model.name, log_path=LOG_PATH+os.sep+model.name)
	model.train(raw_X, y_, sw, dt_config, logger)
	return model


def train_script2():
	from tqdm import tqdm
	from multiprocessing import Pool

	max_leaf_nodes = [None, 2**10, 2**11, 2**12, 2**13, 2**14]
	paras = []
	for leafNum in max_leaf_nodes:
		dt_config = DecisionTreeConfig()
		dt_config.max_leaf_nodes = leafNum
		model_name = 'DecisionTreeModel_01_Ances_leaf{max_leaf_node}'.format(max_leaf_node=dt_config.max_leaf_nodes)
		paras.append((dt_config, model_name))
	with Pool(12) as pool:
		for model in tqdm(pool.imap_unordered(process, paras), total=len(paras), leave=False):
			pass


def test_script(model, logger):
	logger = get_logger(model.name, log_path=LOG_PATH+os.sep+model.name)

	logger.info(model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463']))



train_script2()