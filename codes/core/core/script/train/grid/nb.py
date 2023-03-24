

from core.predict.model_testor import ModelTestor
from core.predict.prob_model.NBModel import MNBConfig, CNBConfig, BNBProbConfig, MNBModel, generate_bnb_prob_model, CNBModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import VEC_TYPE_0_1, TRAIN_MODE
from core.utils.constant import PHELIST_ANCESTOR_DUP, PHELIST_ANCESTOR
from core.helper.data.data_helper import DataHelper

import os
import itertools


def process_mnb(para):
	mnb_config, model_name = para

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR_DUP)
	sw = None

	model = MNBModel(mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, mnb_config, None, save_model=True)

	return model


def train_mnb_script():
	alpha_list = [0.001]
	paras = []
	for alpha in alpha_list:
		mnb_config = MNBConfig()
		mnb_config.alpha = alpha
		model_name = 'MNBModel_alpha{}'.format(alpha)
		paras.append((mnb_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process_mnb(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def process_cnb(para):
	cnb_config, model_name = para
	hpo_reader = HPOReader()

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR_DUP)
	sw = None

	model = CNBModel(mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, cnb_config, None, save_model=False)

	return model


def train_cnb_script():

	alpha_list = [2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
	paras = []
	for alpha in alpha_list:
		cnb_config = CNBConfig()
		cnb_config.alpha = alpha
		model_name = 'CNBModel_alpha{}'.format(alpha)
		paras.append((cnb_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process_cnb(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def process_cnb2(para):
	cnb_config, model_name = para

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	model = CNBModel(vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, cnb_config, None, save_model=True)

	return model


def train_cnb2_script():


	alpha_list = [500.0]
	paras = []
	for alpha in alpha_list:
		cnb_config = CNBConfig()
		cnb_config.alpha = alpha
		model_name = 'CNBModel_01_alpha{}'.format(alpha)
		paras.append((cnb_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process_cnb2(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def process_bnb(para):
	bnb_config, model_name = para
	model = generate_bnb_prob_model(mode=TRAIN_MODE, model_name=model_name)
	model.train(bnb_config, save_model=False)
	return model


def train_bnb_script():


	anno_dp_list = [0.01, 0.05, 0.1, 0.15]
	not_have_dp_list = [0.0, 0.05]
	min_prob_list = [0.05]
	max_prob_list = [0.95]

	paras = []
	for anno_dp, not_have_dp, min_prob, max_prob in itertools.product(anno_dp_list, not_have_dp_list, min_prob_list, max_prob_list):
		bnb_config = BNBProbConfig()
		bnb_config.anno_dp = anno_dp
		bnb_config.not_have_dp = not_have_dp
		bnb_config.min_prob = min_prob
		bnb_config.max_prob = max_prob
		model_name = 'BNBModel_anno_dp{}_not_have_dp{}_min_prob{}_max_prob{}'.format(anno_dp, not_have_dp, min_prob, max_prob)
		paras.append((bnb_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process_bnb(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def train_bnb_script2():
	anno_dp_list = [0.9, 1.0]
	not_have_dp_list = [0.0, 0.1]
	min_max_prob_list = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

	paras = []
	for anno_dp, not_have_dp, minMaxProb in itertools.product(anno_dp_list, not_have_dp_list, min_max_prob_list):
		min_prob, max_prob = minMaxProb
		bnb_config = BNBProbConfig()
		bnb_config.anno_dp = anno_dp
		bnb_config.not_have_dp = not_have_dp
		bnb_config.min_prob = min_prob
		bnb_config.max_prob = max_prob
		model_name = 'BNBModel_anno_dp{}_not_have_dp{}_min_prob{}_max_prob{}'.format(anno_dp, not_have_dp, min_prob, max_prob)
		paras.append((bnb_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process_bnb(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


if __name__ == '__main__':
	train_mnb_script()


	pass