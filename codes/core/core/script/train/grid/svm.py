

import os
from tqdm import tqdm
from multiprocessing import Pool
import itertools

from core.predict.model_testor import ModelTestor
from core.predict.ml_model.SVMModel import generate_model, LSVMConfig
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, DATA_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, LOG_PATH, PREDICT_MODE, TRAIN_MODE, VEC_TYPE_0_1_DIM_REDUCT
from core.utils.constant import PHELIST_ANCESTOR, RESULT_PATH
from core.utils.utils import get_logger
from core.helper.data.data_helper import DataHelper

def process(para):
	lsvm_config, model_name = para

	hpo_reader = HPOReader()
	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, lsvm_config, None, save_model=True)

	return model


def train_script():

	CList = [0.001]
	paras = []
	for C in CList:
		lsvm_config = LSVMConfig()
		lsvm_config.C = C
		model_name = 'LSVMModel_01_Ances_C{}'.format(C)
		paras.append((lsvm_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()

	for para in paras:
		model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, model_name='LSVMModel_01_Ances_C0.001')
		# model = process(para)
		mt.cal_metric_and_save(model, data_names=mt.data_names, use_query_many=True)



if __name__ == '__main__':
	train_script()
	pass
