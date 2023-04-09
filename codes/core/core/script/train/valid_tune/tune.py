import os
import itertools
import numpy as np
from multiprocessing import Pool
import scipy, random

from core.reader.hpo_filter_reader import get_hpo_num_slice_reader, get_IC_slice_reader
from core.reader import HPOReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper, resort_history, combine_history
from core.helper.hyper.para_random_searcher import ParaRandomSearcher
from core.helper.hyper.para_grid_searcher import ParaGridSearcher
from core.predict.model_testor import ModelTestor
from core.utils.constant import VALIDATION_DATA, TEST_DATA, VALIDATION_TEST_DATA, VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_PROB
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP, TEMP_PATH, RESULT_PATH, SEED
from core.utils.constant import CHOOSE_DIS_GEQ_HPO, CHOOSE_DIS_GEQ_IC, get_tune_data_names, get_tune_data_weights, get_tune_metric_names, get_tune_metric_weights
from core.utils.utils import random_string

def resort_history_for_model(hpo_reader_name, save_name):
	for eval_data in [TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA]:
		score_keys = [get_tune_data_names(eval_data), get_tune_metric_names()]
		score_weights = [get_tune_data_weights(eval_data), get_tune_metric_weights()]
		print(score_keys, score_weights)
		resort_history(
			save_folder=os.path.join(RESULT_PATH, hpo_reader_name, 'hyper_tune', save_name, eval_data),
			score_keys=score_keys,
			score_weights=score_weights
		)

def combine_history_for_model(hpo_reader_name, old_save_names, new_save_name,
		score_keys=None, score_weights=None, eval_datas=(TEST_DATA, VALIDATION_DATA), keep_func=None):
	for eval_data_name in eval_datas:
		combine_history(
			[os.path.join(RESULT_PATH, hpo_reader_name, 'hyper_tune', old_save_name, eval_data_name) for old_save_name in old_save_names],
			os.path.join(RESULT_PATH, hpo_reader_name, 'hyper_tune', new_save_name, eval_data_name),
			score_keys=score_keys, score_weights=score_weights, keep_func=keep_func
		)


def get_slice_hpo_reader(choose_dis_mode, min_hpo=None, min_IC=None, slice_phe_list_mode=None):
	if choose_dis_mode == CHOOSE_DIS_GEQ_HPO:
		return get_hpo_num_slice_reader(min_hpo, phe_list_mode=slice_phe_list_mode)
	elif choose_dis_mode == CHOOSE_DIS_GEQ_IC:
		return get_IC_slice_reader(min_IC, phe_list_mode=slice_phe_list_mode)
	else:
		assert False


def flatten_grid(key_to_values):
	"""
	Args:
		key_to_values (dict): {k1: [v1, v2], k2: [v3, v4]}
	Returns:
		list: [{k1: v1, k2: v3}, {k1: v1, k2: v4}, ...]
	"""
	key_list, value_lists = zip(*key_to_values.items())
	return [{k: v for k, v in zip(key_list, v_list)} for v_list in itertools.product(*value_lists)]


def get_valid_score_dict(model, mt, data_names=None, metric_names=None, use_query_many=True, cpu_use=12):
	metric_sets = set(metric_names)
	score_dict = mt.cal_metric_for_multi_data(model, data_names=data_names, metric_set=metric_sets, use_query_many=use_query_many, cpu_use=cpu_use) # {dname: {mname: score}}
	for dname, metric_dict in score_dict.items():
		metric_dict['Mic.RankMedian'] = -metric_dict['Mic.RankMedian']
	return score_dict


def test_model(hyper_helper, model, para_dict, mt, data_names=None, metric_names=None, use_query_many=True, cpu_use=12):
	score_dict = get_valid_score_dict(model, mt, data_names, metric_names, use_query_many, cpu_use)
	hyper_helper.add(para_dict, score_dict=score_dict)
	print(model.name, para_dict, score_dict)
	hyper_helper.save_history()


def train_model_get_score_dict(paras):
	i, train_model_func, para_dict, hpo_reader, model_name, save_model, mt, data_names, metric_names, use_query_many, test_cpu_use = paras
	model = train_model_func(para_dict, hpo_reader=hpo_reader, model_name=model_name, save_model=True, gpu=str(i%2))
	score_dict = get_valid_score_dict(model, mt, data_names, metric_names, use_query_many, test_cpu_use)
	return score_dict, model_name


def train_best_model(para_dict, train_model_func, get_model_func, model_name, hpo_reader=HPOReader(),
		repeat=5, use_query_many=True, cpu_use=2, test_cpu_use=12, use_pool=True, mt_hpo_reader=None,
		eval_data=VALIDATION_DATA):
	def get_new_model_name(model_name, i):
		return model_name + '-v{}-{}'.format(i, random_string(8))

	data_names = get_tune_data_names(eval_data)
	data_weights = get_tune_data_weights(eval_data)
	metric_names = get_tune_metric_names()
	metric_weights = get_tune_metric_weights()

	mt_hpo_reader = mt_hpo_reader or hpo_reader
	mt = ModelTestor(eval_data, hpo_reader=mt_hpo_reader)
	mt.load_test_data(data_names)
	hyper_helper = HyperTuneHelper(
		'TrainBestModel',
		score_keys=[data_names, metric_names],
		score_weights=[data_weights, metric_weights],
	)

	r_model_names = []
	para_list = [
		(i, train_model_func, para_dict, hpo_reader, get_new_model_name(model_name, i), True, mt, data_names, metric_names, use_query_many, test_cpu_use)
		for i in range(repeat)]
	if use_pool:
		with Pool(cpu_use) as pool:
			for score_dict, r_model_name in pool.imap(train_model_get_score_dict, para_list):
				r_model_names.append(r_model_name)
				hyper_helper.add({'r_model_name': r_model_name}, score_dict=score_dict)
				print('{}: {}'.format(r_model_name, score_dict))
	else:
		for para in para_list:
			score_dict, r_model_name = train_model_get_score_dict(para)
			r_model_names.append(r_model_name)
			hyper_helper.add({'r_model_name':r_model_name}, score_dict=score_dict)
			print('{}: {}'.format(r_model_name, score_dict))

	print('Sorted History:', hyper_helper.get_sorted_history())
	model = get_model_func(para_dict, hpo_reader=hpo_reader, model_name=hyper_helper.get_best()['PARAMETER']['r_model_name'], init_para=False)
	model.change_save_folder_and_save(model_name=model_name)

	for r_model_name in r_model_names:
		get_model_func(para_dict,hpo_reader=hpo_reader, model_name=r_model_name, init_para=False).delete_model()
	return model


def get_hyper_dict(hpo_reader_name, eval_data, save_name, save_mode, save_folder):
	data_names = get_tune_data_names(eval_data)
	data_weights = get_tune_data_weights(eval_data)
	metric_names = get_tune_metric_names()
	metric_weights = get_tune_metric_weights()

	save_folder = save_folder or os.path.join(RESULT_PATH, hpo_reader_name, 'hyper_tune', save_name, eval_data)
	hyper_helper = HyperTuneHelper(
		save_name,
		score_keys=[data_names, metric_names],
		score_weights=[data_weights, metric_weights],
		mode=save_mode,
		save_folder= save_folder
	)
	return {
		'data_names': data_names, 'data_weights': data_weights, 'metric_names': metric_names,
		'metric_weights': metric_weights, 'hyper_helper': hyper_helper
	}


def single_train_wrap(paras):
	save_name, para_dict, train_model_func, hpo_reader, hyper_dict, use_query_many, cpu_use, mt_hpo_reader, eval_datas = paras
	print('Begin', save_name, para_dict)
	try:
		model = train_model_func(para_dict, hpo_reader)
	except:
		raise RuntimeError('Can not train with para: {}'.format(para_dict))
	assert hpo_reader.name == model.hpo_reader.name
	eval_to_score_dict = {}
	for eval_data in eval_datas:
		hdict = hyper_dict[eval_data]
		mt = ModelTestor(eval_data, hpo_reader=mt_hpo_reader)
		mt.load_test_data(hdict['data_names'])
		test_model(hdict['hyper_helper'], model, para_dict, mt, data_names=hdict['data_names'],
			metric_names=hdict['metric_names'], use_query_many=use_query_many, cpu_use=cpu_use)
	del model


def tune(grid, train_model_func, save_name, hpo_reader=HPOReader(), save_mode='a', search_type='grid',
		max_iter=None, use_query_many=True, cpu_use=12, hyp_save_folder=None, use_pool=False, mt_hpo_reader=None,
		eval_datas=None):
	"""
	Args:
		grid (dict): {key: value}
		train_model_func (func): paras=(grid,); ret=(Model,)
		search_type (str): 'grid' | 'random'
	"""
	mt_hpo_reader = mt_hpo_reader or hpo_reader
	eval_datas = eval_datas or [VALIDATION_DATA, TEST_DATA]
	hyper_dict = {
		eval_data: get_hyper_dict(hpo_reader.name, eval_data, save_name, save_mode, hyp_save_folder) for eval_data in eval_datas
	}
	for para_dict in get_iterator(grid, search_type, hyper_dict[eval_datas[0]]['hyper_helper'].get_para_history(), max_iter):
		paras = save_name, para_dict, train_model_func, hpo_reader, hyper_dict, use_query_many, cpu_use, mt_hpo_reader, eval_datas
		if use_pool:
			with Pool(1) as pool:
				for _ in pool.imap_unordered(single_train_wrap, [paras]):
					pass
			for eval_data in eval_datas:
				hyper_dict[eval_data]['hyper_helper'].load_history()
		else:
			single_train_wrap(paras)


def multi_train_wrap(para):
	para_dict, hpo_reader, train_model_func = para
	print('Begin', para_dict)
	try:
		model = train_model_func(para_dict, hpo_reader)
	except:
		raise RuntimeError('Can not train with para: {}'.format(para_dict))
	return model, para_dict


def multi_tune(grid, train_model_func, save_name, hpo_reader=HPOReader(), save_mode='a', search_type='grid',
		max_iter=None, use_query_many=True, cpu_use=12, test_cpu_use=12, hyp_save_folder=None, mt_hpo_reader=None,
		eval_datas=None):
	mt_hpo_reader = mt_hpo_reader or hpo_reader
	eval_datas = eval_datas or [VALIDATION_DATA, TEST_DATA]
	hyper_dict = {
		eval_data:get_hyper_dict(hpo_reader.name, eval_data, save_name, save_mode, hyp_save_folder) for eval_data in eval_datas
	}
	with Pool(cpu_use) as pool:
		iterable = get_multi_tune_iterator(grid, search_type, train_model_func, hpo_reader,
			hyper_dict[eval_datas[0]]['hyper_helper'].get_para_history(), max_iter)
		iterable = list(iterable)
		for model, para_dict in pool.imap_unordered(multi_train_wrap, iterable):
			assert hpo_reader.name == model.hpo_reader.name
			for eval_data in eval_datas:
				hdict = hyper_dict[eval_data]
				mt = ModelTestor(eval_data, hpo_reader=mt_hpo_reader)
				mt.load_test_data(hdict['data_names'])
				test_model(hdict['hyper_helper'], model, para_dict, mt, data_names=hdict['data_names'],
					metric_names=hdict['metric_names'], use_query_many=use_query_many, cpu_use=test_cpu_use)


def get_iterator(grid, search_type, history_list=None, max_iter=None):
	if search_type == 'grid':
		return ParaGridSearcher(grid, history_list).iterator()
	elif search_type == 'random':
		return ParaRandomSearcher(grid, history_list).iterator(max_iter)
	assert False


def get_multi_tune_iterator(grid, search_type, train_model_func, hpo_reader, history_list=None, max_iter=None):
	para_iterator = get_iterator(grid, search_type, history_list, max_iter)
	for para_dict in para_iterator:
		yield para_dict, hpo_reader, train_model_func


def get_default_phe_list_mode(vec_type):
	if vec_type == VEC_TYPE_TF_IDF or vec_type == VEC_TYPE_TF:
		return PHELIST_ANCESTOR_DUP
	elif vec_type == VEC_TYPE_0_1 or vec_type == VEC_TYPE_IDF or vec_type == VEC_TYPE_PROB:
		return PHELIST_ANCESTOR
	assert False


def get_default_dtype(vec_type):
	if vec_type == VEC_TYPE_TF_IDF or vec_type == VEC_TYPE_IDF:
		return np.float32
	elif vec_type == VEC_TYPE_0_1 or vec_type == VEC_TYPE_TF:
		return np.int32
	assert False


def get_embed_mat(encoder, **kwargs):
	if encoder == 'deepwalk':
		return get_deep_walk_embed_mat(**kwargs)
	elif encoder == 'glove':
		return get_GloveEmbedMat(**kwargs)
	elif encoder == 'sdne':
		return get_sdne_embed_mat(**kwargs)
	elif encoder == 'gcn':
		return get_gcn_embed_mat(**kwargs)
	elif encoder == 'hce':
		return get_hce_embed_mat(**kwargs)
	assert False


def get_deep_walk_embed_mat(win, numWalk, embed_size):
	encoder_name = 'DeepwalkEncoder_win{}_numwalks{}_embed{}'.format(win, numWalk, embed_size)
	return deepwalk.get_embed(encoder_name)


def get_GloveEmbedMat(embed_size, x_max, phe_list_mode):
	encoder_name = 'GloveEncoder_vec{}_xMax{}_max_iter200'.format(embed_size, x_max)
	return glove.get_embed(encoder_name, phe_list_mode)


def get_sdne_embed_mat(w, lr):
	encoder_name = 'encoder{}_lr{}_epoch400_alpha0.000001_beta5_nu1-0.00001_nu2-0.0001'.format(w, lr)
	return openne.get_embed(encoder_name, 'SDNEEncoder')


def get_gcn_embed_mat(encoder_class, embed_idx, l2_norm, **kwargs):
	if encoder_class == 'GCNDisAsLabelEncoder':
		encoder_name = 'DisAsLabel_xt{}_units{}_lr{}_w_decay{}'.format(kwargs['xtype'], kwargs['units'], kwargs['lr'], kwargs['w_decay'])
	elif encoder_class == 'GCNDisAsLabelFeatureEncoder':
		encoder_name = 'DisAsLabelFeature_layer3_units{}_lr{}_w_decay{}'.format(kwargs['units'], kwargs['lr'], kwargs['w_decay'])
	elif encoder_class == 'GCNDisAsFeatureEncoder':
		encoder_name = 'DisAsFeature_sigmoid_units{}_lr{}_w_decay{}'.format(kwargs['units'], kwargs['lr'], kwargs['w_decay'])
	else:
		assert False
	return gcn.get_embed(encoder_name, encoder_class, embed_idx, l2_norm)


def get_hce_embed_mat(optimizer, bc_type, epoch_num, lr, lambda_, embed_size):
	encoder_name = 'HCEEncoder_{}_{}_epoch{}_lr{}_lambda{}_embedSize{}'.format(optimizer, bc_type, epoch_num, lr, lambda_, embed_size)
	return hce.get_embed(encoder_name)


