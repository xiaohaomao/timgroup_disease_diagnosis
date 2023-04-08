import json
import numpy as np

import codecs

import os, shutil
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import random
import itertools
from copy import deepcopy
from time import time
from collections import Counter
from scipy.stats import wilcoxon, ranksums, mannwhitneyu, shapiro, binom_test

from core.utils.constant import TEST_DATA, VALIDATION_DATA, CUSTOM_DATA, VALIDATION_TEST_DATA, SEED
from core.utils.constant import RESULT_PATH, PHELIST_REDUCE, SORT_P, SORT_P_S, SORT_S_P, DATA_PATH
from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL, DISORDER_GROUP_LEAF_LEVEL
from core.utils.utils import list_find, timer, dict_list_add, unique_list, py_wilcox, import_R, get_all_ancestors, equal, to_rank_scores
from core.utils.utils import cal_boot_conf_int, cal_hodges_lehmann_median_conf_int, cal_portion_conf_int_R, cal_boot_conf_int_for_multi_x
from core.utils.utils import cal_mcnemar_p_value, pvalue_correct, cal_boot_pvalue, dabest_cal_permutation_pvalue, cal_permutation_pvalue
from core.draw.draw import draw_quartile_fig, draw_multi_line_from_df, draw_dodge_bar
from core.draw.simpledraw import simple_line_plot
from core.reader import HPOReader, RDReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader
from core.explainer.explainer import Explainer
from core.helper.data.data_helper import DataHelper
from core.predict.ensemble.rank_score_model import RankScoreModel
from core.predict.ensemble.consistency_rate_model import ConsistencyRateModel


class ModelTestor(object):
	def __init__(self, eval_data=TEST_DATA, hpo_reader=HPOReader(), seed=None, keep_general_dis_map=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
			eval_data (str or None): 'test' | 'validation' | 'validation_test' | 'custom'
		"""
		print('HPOReader of ModelTestor:', hpo_reader.name)
		self.hpo_reader = hpo_reader
		self.rd_reader = None
		integrate_prefix = 'INTEGRATE_'
		self.use_rd_code = self.hpo_reader.name.startswith(integrate_prefix)
		self.keep_general_dis_map = keep_general_dis_map
		self.top_n_list = (1, 3, 5, 10, 20, 30, 40, 50, 100, 200)
		self.bar_metrics = {
			'MicroF1', 'MacroF1', 'MicroPrecision', 'MacroPrecision', 'MicroRecall', 'MacroRecall',
			'Mic.RankMedian', 'Mac.RankMedian',
			'Mic.Recall.1', 'Mic.Recall.3', 'Mic.Recall.5', 'Mic.Recall.10', 'Mic.Recall.20', 'Mic.Recall.30', 'Mic.Recall.40', 'Mic.Recall.50', 'Mic.Recall.100', 'Mic.Recall.200',
			'Mac.Recall.1', 'Mac.Recall.3', 'Mac.Recall.5', 'Mac.Recall.10', 'Mac.Recall.20', 'Mac.Recall.30', 'Mac.Recall.40', 'Mac.Recall.50', 'Mac.Recall.100', 'Mac.Recall.200',
		}
		self.quartile_metrics = {
			'Rank'
		}
		self.all_metrics = self.bar_metrics | self.quartile_metrics
		self.RESULT_PATH = os.path.join(RESULT_PATH, self.get_hpo_reader().name)
		self.RESULT_FIG_PATH = os.path.join(self.RESULT_PATH, 'fig')
		self.DELETE_PATH = os.path.join(self.RESULT_PATH, 'delete')
		os.makedirs(self.DELETE_PATH, exist_ok=True)
		self.CSV_FOLDER_PATH = os.path.join(self.RESULT_PATH, 'csv')
		os.makedirs(self.CSV_FOLDER_PATH, exist_ok=True)
		self.init_data_attr(eval_data, hpo_reader)
		self.random = random.Random(seed)


	def init_data_attr(self, eval_data, hpo_reader):
		self.eval_data = eval_data
		self.dh = DataHelper(hpo_reader)
		self.data = {}
		if eval_data == TEST_DATA or eval_data == VALIDATION_TEST_DATA:
			self.data_names = self.dh.test_names
		elif eval_data == VALIDATION_DATA:
			self.data_names = self.dh.valid_names
		elif eval_data == CUSTOM_DATA:
			self.data_names = []
			self.name_to_path = {}
		else:
			assert False


	def process_sheet_name(self, sheet_name):
		if len(sheet_name) > 31:
			return sheet_name[:31]
		return sheet_name


	def get_phenomizer_test_data(self, data_name):
		dh = DataHelper(hpo_reader=HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD']))
		patients = dh.get_dataset(data_name, self.eval_data, filter=False)
		if not self.keep_general_dis_map:
			patients = [[hpo_list, self.remove_dis_map_general(dis_list, use_rd_code=False)] for hpo_list, dis_list in patients]
		return patients


	def load_test_data(self, data_names=None):
		if data_names is None:
			data_names = self.data_names
		if self.eval_data == CUSTOM_DATA:
			for dname in data_names:
				self.data[dname] = self.dh.get_dataset_with_path(self.name_to_path[dname])
		elif self.eval_data == VALIDATION_TEST_DATA:
			for dname in data_names:
				self.data[dname] = self.dh.get_dataset(dname, VALIDATION_DATA, filter=False) + self.dh.get_dataset(dname, TEST_DATA, filter=False)


		else:
			assert self.eval_data == TEST_DATA or self.eval_data == VALIDATION_DATA
			for dname in data_names:

				self.data[dname] = self.dh.get_dataset(dname, self.eval_data, filter=False)


		if not self.keep_general_dis_map:
			for data_name in data_names:
				self.data[data_name] = [[hpo_list, self.remove_dis_map_general(dis_list)] for hpo_list, dis_list in self.data[data_name]]


	def get_dataset(self, data_name):
		if data_name not in self.data:
			self.load_test_data([data_name])
		return self.data[data_name]


	def get_dataset_size(self, data_name):
		return len(self.get_dataset(data_name))


	def set_custom_data_set(self, name_to_path, data_names=None):
		assert self.eval_data == CUSTOM_DATA
		self.data_names = list(name_to_path.keys()) if data_names is None else data_names
		self.name_to_path = name_to_path


	def get_hpo_reader(self):
		if self.hpo_reader is None:
			self.hpo_reader = HPOReader()

		return self.hpo_reader


	def get_rd_reader(self):
		if self.rd_reader is None:
			self.rd_reader = RDReader()
		return self.rd_reader

	def get_eval_data_path(self, eval_data=None):
		return os.path.join(self.RESULT_PATH, 'Metric-{}'.format(eval_data or self.eval_data))

	def get_metric_path(self, metric_name):
		return os.path.join(self.get_eval_data_path(self.eval_data), metric_name)

	def get_metric_model_path(self, metric_name, model_name):
		return os.path.join(self.get_metric_path(metric_name), model_name)

	def get_result_file_path(self, metric_name, model_name, data_name, postfix='.json'):
		return os.path.join(self.get_metric_model_path(metric_name, model_name), data_name+postfix)

	def get_raw_result_path(self, model_name, data_name, postfix='.npz'):
		return os.path.join(self.RESULT_PATH, 'RawResults', model_name, '{}-{}{}'.format(self.eval_data, data_name, postfix))

	def get_default_metric_change_with_hpo_num_fig_path(self, metric_name, data_name, target='patient'):
		return os.path.join(self.RESULT_FIG_PATH, 'HPO-Num', '{}-{}-{}-{}.png'.format(self.eval_data, metric_name, data_name, target))

	def get_default_pa_num_with_hpo_num_fig_path(self, data_name, target='patient'):
		return os.path.join(self.RESULT_FIG_PATH, 'HPO-Num', '{}-{}-{}-{}.png'.format('1', self.eval_data, data_name, target))

	def get_table_result_folder(self):
		return os.path.join(self.RESULT_PATH, 'table')

	def get_dis_category_result_folder(self):
		return os.path.join(self.RESULT_PATH, 'DisCategoryResult')

	def get_case_result_folder(self):
		return os.path.join(self.RESULT_PATH, 'CaseResult')

	def get_metric_bar_plot_fig_folder(self):
		return os.path.join(self.RESULT_FIG_PATH, 'Barplot')

	def get_metirc_quartile_plot_fig_folder(self):
		return os.path.join(self.RESULT_FIG_PATH, 'QuartilePlot')

	def data_names_to_key(self, data_names):
		return '-'.join(sorted(data_names))

	def get_model_name_with_mark(self, model_name, mark):
		return f'{model_name}-{mark}'

	@timer
	def save_raw_results(self, raw_results, model_name, data_name):

		dis_map_rank = self.get_hpo_reader().get_dis_map_rank()











		file_path = self.get_raw_result_path(model_name, data_name, '.npz')
		dir_name = os.path.dirname(file_path); os.makedirs(dir_name, exist_ok=True)
		row_num, col_num = len(raw_results), len(raw_results[0])
		dis_code_mat = np.array([ [dis_map_rank[raw_results[i][j][0]] for j in range(col_num)] for i in range(row_num)], dtype=np.int32)


		score_mat = np.array([ [raw_results[i][j][1] for j in range(col_num)] for i in range(row_num)], dtype=np.float32)

		np.savez(file_path, dis_code_mat=dis_code_mat, score_mat=score_mat)




	@timer
	def load_raw_results(self, model_name, data_name):
		"""
		Returns:
			list: [[(dis1, score1), ...], ...]
		"""
		if model_name == 'Phenomizer':
			raw_results, _ = self.get_phenomizer_raw_results_and_patients(data_name)
			return raw_results
		if model_name.endswith('INTEGRATE_CCRD_OMIM_ORPHA'):
			pass

		elif model_name.endswith('CCRD_OMIM_ORPHA'):
			model_name = model_name[0: model_name.find('-CCRD_OMIM_ORPHA')]




		dis_list = self.get_hpo_reader().get_dis_list()









		file_path = self.get_raw_result_path(model_name, data_name, '.npz')




		loaded = np.load(file_path)
		dis_code_mat, score_mat = loaded['dis_code_mat'], loaded['score_mat']


		row_num, col_num = dis_code_mat.shape
		raw_results = [ [(dis_list[dis_code_mat[i][j]], score_mat[i][j]) for j in range(col_num)] for i in range(row_num)] # 速度瓶颈; decode将byte转为str


		return raw_results


	def exist_raw_results(self, model_name, data_name):
		return os.path.exists(self.get_raw_result_path(model_name, data_name, '.npz'))


	def save_metric_dict(self, model_name, data_name, metric_dict):
		for metric_name, metric_result in metric_dict.items():
			dir_path = self.get_metric_model_path(metric_name, model_name)
			json_path = self.get_result_file_path(metric_name, model_name, data_name, '.json')
			os.makedirs(dir_path, exist_ok=True)
			json.dump({
				'Model': model_name,
				'Dataset': data_name,
				metric_name: metric_result
			}, open(json_path, 'w'), indent=2)


	def cal_quartile(self, numList):
		"""
		Args:
			data_list (list): list of number
		Returns:
			list: [minimum, Q1, median, Q3, maximum]
		"""
		return np.percentile(numList, [0, 25, 50, 75, 100])


	def cal_all_metric_multi_run_func(self, paras):
		model, data_item = paras
		patient, dis_codes = data_item
		raw_result = model.query(patient, topk=None)
		return raw_result


	def get_dis_rank(self, query_result, dis_codes, rd_decompose=False):

		if len(dis_codes) == 1:
			rank = list_find(query_result, lambda x: x[0]==dis_codes[0])
		else:
			dis_code = set(dis_codes)
			rank = list_find(query_result, lambda x: x[0] in dis_code)
		if rank != -1 and rd_decompose:
			rd_to_sources = self.get_rd_reader().get_rd_to_sources()
			ret_rank = 0
			for i in range(rank):
				ret_rank += len(rd_to_sources[query_result[i][0]])
			return ret_rank
		return rank


	def get_ytrue(self, y_pred_code, y_true_codes):
		"""
		Args:
			y_pred_code (str)
			y_true_codes (list)
		Returns:
			str
		"""

		i = list_find(y_true_codes, lambda code: code==y_pred_code)

		if i < 0:





			return self.random.choice(y_true_codes)
		return y_pred_code


	def result_to_important(self, raw_result, dis_codes, rd_decompose=False):
		rank = self.get_dis_rank(raw_result, dis_codes, rd_decompose=rd_decompose)


		if rank == -1:
			keep_dnames = ['OMIM', 'ORPHA', 'CCRD']
			dis_num = HPOIntegratedDatasetReader(keep_dnames=keep_dnames).get_dis_num() if self.use_rd_code or not rd_decompose else HPOFilterDatasetReader(keep_dnames=keep_dnames)
			rank = int((dis_num - 1) / 2)
		top_n_hit = [0]*len(self.top_n_list) if rank < 0 else [1 if rank < top_n else 0 for top_n in self.top_n_list]
		predict = raw_result[0] # (dis_codes, score)
		return rank+1, top_n_hit, predict


	def result_to_important_wrap(self, para):
		raw_result, dis_codes, rd_decompose = para
		return self.result_to_important(raw_result, dis_codes, rd_decompose)


	@timer
	def raw_results_to_importants(self, raw_results, test_patients, cpu_use, chunk_size, rd_decompose=False):
		data_size = len(raw_results)
		paras = [(raw_results[i], test_patients[i][1], rd_decompose) for i in range(data_size)]
		rank_list, top_n_hit_lists, pred_dis_and_score_list = [], [], []
		if cpu_use == 1:
			return zip(*[self.result_to_important_wrap(para) for para in paras])
		else:
			with Pool(cpu_use) as pool:
				for rank, top_n_hit, pred_dis_and_score in tqdm(pool.imap(self.result_to_important_wrap, paras, chunksize=chunk_size), total=len(paras), leave=False):
					rank_list.append(rank)
					top_n_hit_lists.append(top_n_hit)
					pred_dis_and_score_list.append(pred_dis_and_score)
		return rank_list, top_n_hit_lists, pred_dis_and_score_list


	def check_change_score(self, raw_results):
		score_type = type(raw_results[0][0][1])
		if score_type == int or score_type == float:
			return
		for i in range(len(raw_results)):
			raw_results[i] = [(dis_code, float(score)) for dis_code, score in raw_results[i]]


	def dis_codes_to_id(self, dis_codes):
		return str(sorted(dis_codes))


	def get_conf_int(self, raw_results, test_data, metric_set, chunk_size=20, conf_level=0.95, rd_decompose=False):
		"""
		Returns:
			dict: {metric_name: (low, high)}
		"""
		rank_list, top_n_hit_lists, pred_dis_and_score_list = self.raw_results_to_importants(raw_results, test_data, 1, chunk_size, rd_decompose)
		rank_array, top_n_hit_matrix = np.array(rank_list), np.array(top_n_hit_lists)
		ret_dict = {}
		if 'Mic.RankMedian' in metric_set:
			ret_dict['Mic.RankMedian'] = self.cal_conf_interval(rank_array, metric_type='median', conf_level=conf_level)

		dis_codes_to_idx_list = {}
		for i, pa in enumerate(test_data):
			dict_list_add(self.dis_codes_to_id(pa[1]), i, dis_codes_to_idx_list)
		for i in range(len(self.top_n_list)):
			k = 'Mic.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = self.cal_conf_interval(top_n_hit_matrix[:, i], metric_type='mean', conf_level=conf_level)
			k = 'Mac.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = self.cal_conf_interval(top_n_hit_matrix[:, i], metric_type='mac_mean',
					id_lists=list(dis_codes_to_idx_list.values()), conf_level=conf_level)
		return ret_dict


	def get_multi_data_mean_conf_int(self, multi_raw_results, multi_test_data, metric_set, chunk_size=20, conf_level=0.95, rd_decompose=False):
		multi_rank_arys, multi_top_n_hit_matrices = [], []
		for raw_results, test_data in zip(multi_raw_results, multi_test_data):
			rank_list, top_n_hit_lists, pred_dis_and_score_list = self.raw_results_to_importants(raw_results, test_data, 1, chunk_size, rd_decompose)
			multi_rank_arys.append(np.array(rank_list))
			multi_top_n_hit_matrices.append(np.array(top_n_hit_lists))
		ret_dict = {}
		if 'Mic.RankMedian' in metric_set:
			ret_dict['Mic.RankMedian'] = self.cal_multi_x_mean_stat_conf_interval(
				multi_rank_arys, metric_type='median', conf_level=conf_level)
		multi_id_lists = []
		for test_data in multi_test_data:
			dis_codes_to_idx_list = {}
			for i, pa in enumerate(test_data):
				dict_list_add(self.dis_codes_to_id(pa[1]), i, dis_codes_to_idx_list)
			multi_id_lists.append(list(dis_codes_to_idx_list.values()))
		for i in range(len(self.top_n_list)):
			k = 'Mic.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = self.cal_multi_x_mean_stat_conf_interval(
					[top_n_hit_matrix[:, i] for top_n_hit_matrix in multi_top_n_hit_matrices], metric_type='mean', conf_level=conf_level)
			k = 'Mac.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = self.cal_multi_x_mean_stat_conf_interval(
					[top_n_hit_matrix[:, i] for top_n_hit_matrix in multi_top_n_hit_matrices], metric_type='mac_mean',
					multi_id_lists=multi_id_lists, conf_level=conf_level)
		return ret_dict


	@timer
	def get_performance(self, raw_results, test_data, metric_set, logger=None, cpu_use=12, chunk_size=20, rd_decompose=False):
		"""
		Args:
			raw_results (list): [[(dis1, score1), ...], ...]
			test_data (list): [(hpo_codes, dis_codes), ...];
			metric_set (set)
		Returns:
			dict: {'TopAcc.10': 3, ...}
		"""

		if len(raw_results) == 0:
			return {metric: np.nan for metric in metric_set}




		rank_list, top_n_hit_lists, pred_dis_and_score_list = self.raw_results_to_importants(raw_results, test_data, 1, chunk_size, rd_decompose)
		rank_array, top_n_hit_matrix = np.array(rank_list), np.array(top_n_hit_lists)
		ret_dict, output_str = {}, ''
		rank_quatile = self.cal_quartile(rank_array)
		output_str += '[minimum, Q1, median, Q3, maximum] = [%.2f, %.2f, %.2f, %.2f, %.2f]\n' % tuple(rank_quatile)

		if 'Mic.RankMedian' in metric_set:
			ret_dict['Mic.RankMedian'] = rank_quatile[2]
		mic_recall_n = top_n_hit_matrix.mean(axis=0)
		for i in range(len(self.top_n_list)):
			k = 'Mic.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = mic_recall_n[i]

		dis_codes_to_idx_list = {}
		for i, pa in enumerate(test_data):
			dict_list_add(self.dis_codes_to_id(pa[1]), i, dis_codes_to_idx_list)

		mac_recall_n = np.array([top_n_hit_matrix[idx_list].mean(axis=0) for idx_list in dis_codes_to_idx_list.values()]).mean(axis=0)
		for i in range(len(self.top_n_list)):
			k = 'Mac.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = mac_recall_n[i]

		y_pred = [dis_code for dis_code, score in pred_dis_and_score_list]
		y_true = [self.get_ytrue(y_pred[i], test_data[i][1]) for i in range(len(y_pred))]
		if 'MicroF1' in metric_set:
			ret_dict['MicroF1'] = f1_score(y_true, y_pred, average='micro')
		if 'MacroF1' in metric_set:
			ret_dict['MacroF1'] = f1_score(y_true, y_pred, average='macro')
		if 'MicroPrecision' in metric_set:
			ret_dict['MicroPrecision'] = precision_score(y_true, y_pred, average='micro')
		if 'MacroPrecision' in metric_set:
			ret_dict['MacroPrecision'] = precision_score(y_true, y_pred, average='macro')
		if 'MicroRecall' in metric_set:
			ret_dict['MicroRecall'] = recall_score(y_true, y_pred, average='micro')
		if 'MacroRecall' in metric_set:
			ret_dict['MacroRecall'] = recall_score(y_true, y_pred, average='macro')
		for ret_item in ret_dict.items():
			output_str += '%s: %s\n' % ret_item
		if 'Rank' in metric_set:
			ret_dict['Rank'] = rank_list
		if logger is not None:
			logger.info(output_str)
		return ret_dict


	def _cal_metric(self, model, data_name, cpu_use, use_query_many, chunk_size=None,
			save_raw_results=False, metric_set=None, logger=None, save_model_name=None, rd_decompose=False):
		if logger is not None:
			logger.info('\n------------------------------\n%s, %s (SIZE=%d), calculating Metrics:' % (
				save_model_name or model.name, data_name, len(self.data[data_name])))
		metric_set = metric_set or self.all_metrics


		chunk_size = self.get_chunk_size(self.get_dataset_size(data_name), cpu_use) if chunk_size is None else chunk_size
		raw_results = []
		if use_query_many:
			patients, dis_codes = list(zip(*self.data[data_name]))


			raw_results = model.query_many(patients, topk=None, cpu_use=cpu_use, chunk_size=chunk_size)


		else:
			if cpu_use == 1: # single thread
				for data_item in tqdm(self.data[data_name]):
					raw_result = self.cal_all_metric_multi_run_func( (model, data_item) )
					raw_results.append(raw_result)
			else:   # multi thread
				para_list = [(model, data_item) for data_item in self.data[data_name]]
				with Pool(cpu_use) as pool:
					for raw_result in tqdm(pool.imap(self.cal_all_metric_multi_run_func, para_list, chunksize=chunk_size), total=len(para_list), leave=False):
						raw_results.append(raw_result)
		if save_raw_results:
			self.save_raw_results(raw_results, save_model_name or model.name, data_name)

		ret_dict = self.get_performance(raw_results, self.data[data_name], metric_set, logger, cpu_use=cpu_use, rd_decompose=rd_decompose)


		return ret_dict


	def cal_and_save_performance_from_raw_results(self, model_name, data_name, metric_set=None, logger=None, rd_decompose=False):

		metric_set = metric_set or self.all_metrics
		raw_results = self.load_raw_results(model_name, data_name)


		if data_name not in self.data:
			self.load_test_data([data_name])

		metric_dict = self.get_performance(raw_results, self.data[data_name], metric_set, logger=logger, rd_decompose=rd_decompose)
		self.save_metric_dict(model_name, data_name, metric_dict)


	def cal_mean_metric_over_datasets(self, model, metric_names, data_names=None, cpu_use=12, use_query_many=True, chunk_size=None, logger=None):
		"""
		Returns:
			list: [metric1, metric2, metric3, ...]
		"""
		if data_names is None:
			data_names = self.data_names
		dname_to_metric = self.cal_metric_for_multi_data(model, data_names, set(metric_names), cpu_use, use_query_many, chunk_size, logger=logger)
		return self.cal_mean_metric(dname_to_metric, metric_names, data_names)


	def cal_mean_metric(self, dname_to_metric, metric_names, data_names):
		return list(np.mean([[dname_to_metric[dname][metric] for dname in data_names] for metric in metric_names], axis=1))


	def cal_metric_for_multi_data(self, model, data_names=None, metric_set=None, cpu_use=12, use_query_many=True,
			chunk_size=None, save_raw_results=False, logger=None):
		"""
		Returns:
			dict {data_names: metric_dict}
		"""
		if data_names is None:
			data_names = self.data_names
		ret_dict = {}
		for data_name in data_names:
			ret_dict[data_name] = self._cal_metric(model, data_name, cpu_use, use_query_many,
				chunk_size, save_raw_results, metric_set, logger)
		return ret_dict


	def cal_metric_and_save(self, model, data_names=None, metric_set=None, cpu_use=12,
			use_query_many=True, chunk_size=None, save_raw_results=False, logger=None,
			save_model_name=None, rd_decompose=False):
		if data_names is None:
			data_names = self.data_names
		ret_dict = {}


		for data_name in data_names:
			metric_dict = self._cal_metric(model, data_name, cpu_use, use_query_many,
				chunk_size, save_raw_results, metric_set, logger, save_model_name=save_model_name, rd_decompose=rd_decompose)
			self.save_metric_dict(save_model_name or model.name, data_name, metric_dict)
			ret_dict[data_name] = metric_dict

		return ret_dict


	def cal_metric_conf_int_and_save(self, model_name, data_names=None, conf_level=0.95, metric_set=None,
			cal_multi=True, cal_single=True, rd_decompose=False):
		data_names = data_names or self.data_names
		metric_names = list(metric_set)
		for dname in data_names:
			if dname not in self.data:
				self.load_test_data([dname])
		multi_raw_results = [self.load_raw_results(model_name, data_name) for data_name in data_names]

		if cal_single:
			for data_name, raw_results in zip(data_names, multi_raw_results):

				test_data = self.data[data_name] if model_name != 'Phenomizer' else self.get_phenomizer_test_data(data_name)
				metric_dict = self.get_conf_int(raw_results, test_data, metric_set, rd_decompose=rd_decompose)
				#print(metric_dict)
				for metric_name in metric_names:
					save_json = self.get_result_file_path(metric_name, model_name, data_name)
					save_dict = json.load(open(save_json))
					save_dict[f'CONF_INT_{conf_level}'] = metric_dict[metric_name]
					json.dump(save_dict, open(save_json, 'w'), indent=2)
		if cal_multi:
			data_name = self.data_names_to_key(data_names)
			print('calculating confidence interval: ', model_name, data_name)
			test_datas = [self.data[data_name] if model_name != 'Phenomizer' else self.get_phenomizer_test_data(data_name) for data_name in data_names]
			metric_dict = self.get_multi_data_mean_conf_int(
				multi_raw_results, test_datas, metric_set, conf_level=conf_level, rd_decompose=rd_decompose)
			for metric_name in metric_set:
				save_json = self.get_result_file_path(metric_name, model_name, data_name)
				json.dump({
					'Model': model_name,
					'Dataset': data_name,
					f'CONF_INT_{conf_level}': metric_dict[metric_name]
				}, open(save_json, 'w'), indent=2)


	def cal_metric_conf_int_and_save_wrap(self, paras):
		return self.cal_metric_conf_int_and_save(*paras)


	def cal_metric_conf_int_and_save_parallel(self, model_names, data_names=None,
			conf_level=0.95, metric_set=None, cal_multi=True, cal_single=True, cpu_use=12, rd_decompose=False):
		def get_iterator():
			for model_name in model_names:
				yield model_name, data_names, conf_level, metric_set, cal_multi, cal_single, rd_decompose
		with Pool(min(len(model_names), cpu_use)) as pool:
			pool.map(self.cal_metric_conf_int_and_save_wrap, get_iterator())


	def get_chunk_size(self, datasize, cpu_use):
		return max(min(int(datasize/cpu_use), 200), 20)


	def call_all_metric_and_save(self, model, data_name, cpu_use=os.cpu_count(), use_query_many=False, chunk_size=200, save_raw_results=False):
		self.cal_metric_and_save(model, [data_name], None, cpu_use, use_query_many, chunk_size, save_raw_results)


	def result_to_presult_multi_wrap(self, paras):
		pmodel, phe_list, result = paras
		return pmodel.result_to_presult(phe_list, result)


	def cal_pvalue_metric_and_save(self, pmodel, data_names, metric_set, sort_types=None, logger=None, save_raw_results=False):
		if sort_types is None:
			sort_types = [pmodel.sort_type]
		for data_name in data_names:
			#print(data_name)
			raw_results = self.load_raw_results(pmodel.model.name, data_name)
			data = self.get_dataset(data_name)
			phe_lists, _ = zip(*data)
			qscore_vecs = [pmodel.result_to_query_score_vec(result) for result in raw_results]
			ps = [pmodel.q_score_vec_to_pvalue(phe_list, qscore_vec) for phe_list, qscore_vec in tqdm(zip(phe_lists, qscore_vecs), total=len(data))]
			for sort_type in sort_types:
				pmodel.set_sort_type(sort_type)

				praw_results = [pmodel.sort_result(pmodel.dis_list, qscore_vec, p, pmodel.sort_type, topk=None) for qscore_vec, p in zip(qscore_vecs, ps)]
				metric_dict = self.get_performance(praw_results, data, metric_set, logger)
				self.save_metric_dict(pmodel.name, data_name, metric_dict)
				if save_raw_results:
					self.save_raw_results(praw_results, pmodel.name, data_name)


	def cal_metric_dict_with_condition(self, test_data, raw_results, choose_pa_func, metric_set=None):
		"""
		Args:
			test_data (list): [[hpo_list, dis_list], ...]
			raw_results (list): [[(dis_code, score), ...], ]
			choose_pa_func (func): func(idx, patient)
		Returns:
			dict: {'TopAcc.10': 3, ...}
		"""
		choose_pa_ranks = [i for i in range(len(test_data)) if choose_pa_func(i, test_data[i])]
		raw_results = [raw_results[rank] for rank in choose_pa_ranks]
		test_data = [test_data[rank] for rank in choose_pa_ranks]
		metric_dict = self.get_performance(raw_results, test_data, metric_set)
		return metric_dict


	def cal_change_with_hpo_length(self, model_name, data_name, metric_set, x_list, target='patient'):
		"""
		Args:
			target (str): 'patient' | 'disease'
		Returns:
			dict: {metric_name1: score_list, metric_name2: score_list}; len(score_list) -> [min_hpo, max_hpo)
		"""
		def get_choose_pa_func(i):
			if target == 'patient':
				return lambda idx, pa_item:len(pa_item[0]) >= i
			elif target == 'disease':
				return lambda idx, pa_item:dis_len_list[idx] >= i
			else:
				assert False
		ret_dict = {metric_name: [] for metric_name in metric_set}
		raw_results = self.load_raw_results(model_name, data_name)
		data = self.get_dataset(data_name)
		if target == 'disease':
			dis2hpo = self.get_hpo_reader().get_dis_to_hpo_dict(PHELIST_REDUCE)
			dis_len_list = [np.median([len(dis2hpo[dis_code]) for dis_code in pa_item[1]]) for pa_item in data]
		for i in x_list:
			choose_pa_func = get_choose_pa_func(i)
			metric_dict = self.cal_metric_dict_with_condition(data, raw_results, choose_pa_func, metric_set)
			for metric_name in metric_set:
				ret_dict[metric_name].append(metric_dict[metric_name])
		return ret_dict


	def cal_change_with_hpo_length_wrapper(self, paras):
		model_name, data_name, metric_set, x_list, target = paras
		return model_name, self.cal_change_with_hpo_length(model_name, data_name, metric_set, x_list, target=target)


	def draw_change_with_hpo_length(self, data_name, model_names, metric_set, cpu_use=None, min_hpo=1, max_hpo=31, target='patient'):
		if cpu_use is None:
			cpu_use = min(12, len(model_names))
		x_list = list(range(min_hpo, max_hpo))
		metric_to_model_to_score_list = {metric_name:{} for metric_name in metric_set}
		with Pool(cpu_use) as pool:
			para_list = [(model_name, data_name, metric_set, x_list, target) for model_name in model_names]
			for model_name, metric_to_slist in tqdm(pool.imap_unordered(self.cal_change_with_hpo_length_wrapper, para_list), total=len(para_list), leave=False):
				for metric, score_list in metric_to_slist.items():
					metric_to_model_to_score_list[metric][model_name] = score_list
		for metric_name, modelToScoreList in metric_to_model_to_score_list.items():
			self._draw_change_with_hpo_length(data_name, metric_name, model_names, modelToScoreList, x_list, target=target)


	def _draw_change_with_hpo_length(self, data_name, metric_name, model_names, modelToScoreList, x_list, target, fig_path=None):
		if fig_path is None:
			fig_path = self.get_default_metric_change_with_hpo_num_fig_path(metric_name, data_name, target=target)
		columns = ['HPO_GEQ_N', metric_name, 'Model']
		df_dict = {col: [] for col in columns}
		for model_name, score_list in modelToScoreList.items():
			df_dict['HPO_GEQ_N'].extend(x_list)
			df_dict[metric_name].extend(score_list)
			df_dict['Model'].extend([model_name]*len(score_list))
		draw_multi_line_from_df(
			pd.DataFrame(df_dict, columns=columns), fig_path, x_col='HPO_GEQ_N', y_col=metric_name,
			class_col='Model', class_order=model_names
		)


	def draw_patient_num_with_hpo_length(self, data_name, min_hpo=1, max_hpo=31, target='patient', fig_path=None):
		if fig_path is None:
			fig_path = self.get_default_pa_num_with_hpo_num_fig_path(data_name, target=target)
		dataset = self.get_dataset(data_name)
		if target == 'patient':
			counter = Counter([len(hpo_list) for hpo_list, dis_list in dataset])
		elif target == 'disease':
			dis2hpo = self.get_hpo_reader().get_dis_to_hpo_dict(PHELIST_REDUCE)
			dis_len_list = [np.median([len(dis2hpo[dis_code]) for dis_code in pa_item[1]]) for pa_item in dataset]
			counter = Counter(dis_len_list)
		else:
			assert False
		total = sum([pNum for hpo_num, pNum in counter.items() if hpo_num >= max_hpo])
		x_list = range(max_hpo - 1, min_hpo - 1, -1)
		y_list = []
		for i in x_list:
			total += counter.get(i, 0)
			y_list.append(total)
		simple_line_plot(fig_path, x_list, y_list, 'HPO_GEQ_N', 'Patient Number', )


	def cal_dis_category_result(self, data_name, model_name, metric_set, dis_codes_id_set=None):
		"""
		Returns:
			dict: {metric_name: {dis_code: score}}
		"""
		dataset = self.get_dataset(data_name)
		raw_results = self.load_raw_results(model_name, data_name)
		dis_codes_id_set = {self.dis_codes_to_id(dis_list) for _, dis_list in dataset} if dis_codes_id_set is None else dis_codes_id_set
		ret_dict = {metric_name: {} for metric_name in metric_set}
		for dis_codes_id in dis_codes_id_set:
			choose_pa_func = lambda idx, pa_item: dis_codes_id == self.dis_codes_to_id(pa_item[1])
			choose_pa_ranks = [i for i in range(len(dataset)) if choose_pa_func(i, dataset[i])]
			dis_raw_results = [raw_results[rank] for rank in choose_pa_ranks]
			test_data = [dataset[rank] for rank in choose_pa_ranks]
			metric_dict = self.get_performance(dis_raw_results, test_data, metric_set)
			for metric_name, score in metric_dict.items():
				ret_dict[metric_name][dis_codes_id] = score
		return ret_dict


	def cal_dis_category_result_wrapper(self, paras):
		data_name, model_name, metric_set, dis_codes_id_set = paras
		return model_name, self.cal_dis_category_result(data_name, model_name, metric_set, dis_codes_id_set)


	def gen_dis_category_result_xlsx(self, data_name, model_names, metric_set, cpu_use=None, folder=None, reverse=False):
		def to_df_dict(model_name, dis_codes_id_to_score, dis_order):
			return {model_name: [dis_codes_id_to_score[dis_codes_id] for dis_codes_id in dis_order]}
		if cpu_use is None: cpu_use = min(12, len(model_names))
		if folder is None: folder = self.get_dis_category_result_folder()
		os.makedirs(folder, exist_ok=True)

		dataset = self.get_dataset(data_name)
		dis_codes_id_list = [self.dis_codes_to_id(dis_list) for _, dis_list in dataset]
		dis_codes_id_to_dis_list = {dis_codes_id_list[i]: dataset[i][1] for i in range(len(dis_codes_id_list))}
		dis_codes_id_set = set(dis_codes_id_list)
		dis_order = list(dis_codes_id_set)
		metric_to_df_dict = {metric: {} for metric in metric_set}
		with Pool(cpu_use) as pool:
			para_list = [(data_name, model_name, metric_set, dis_codes_id_set) for model_name in model_names]
			for model_name, metric_to_dis_to_score in tqdm(pool.imap_unordered(self.cal_dis_category_result_wrapper, para_list), total=len(para_list), leave=False):
				for metric, dis_codes_id_to_score in metric_to_dis_to_score.items():
					metric_to_df_dict[metric].update(to_df_dict(model_name, dis_codes_id_to_score, dis_order))

		dis_codes_id_to_pa_num = Counter(dis_codes_id_list)

		dis_codes_id_to_ave_pa_hpo_num = {}
		for i, dis_codes_id in enumerate(dis_codes_id_list):
			dict_list_add(dis_codes_id, len(dataset[i][0]), dis_codes_id_to_ave_pa_hpo_num)
		dis_codes_id_to_ave_pa_hpo_num = {dis_codes_id: np.mean(hpo_lenList) for dis_codes_id, hpo_lenList in dis_codes_id_to_ave_pa_hpo_num.items()}

		dis2hpo = self.get_hpo_reader().get_dis_to_hpo_dict(PHELIST_REDUCE)
		dis_codes_id_to_ave_dis_hpo_num = {dis_codes_id: np.mean([len(dis2hpo[dis_code]) for dis_code in dis_codes])
			for dis_codes_id, dis_codes in dis_codes_id_to_dis_list.items()}

		explainer = Explainer()
		columns = ['DISEASE_CODE', 'DISEASE_NAME', 'PATIENT_NUM', 'DISEASE_HPO_NUM', 'PATIENT_HPO_AVE_NUM'] + model_names
		for metric, df_dict in metric_to_df_dict.items():
			df_dict['DISEASE_CODE'] = dis_order
			df_dict['DISEASE_NAME'] = explainer.add_cns_info([eval(dis_codes_id) for dis_codes_id in dis_order])
			df_dict['PATIENT_NUM'] = [dis_codes_id_to_pa_num[dis_codes_id] for dis_codes_id in dis_order]
			df_dict['DISEASE_HPO_NUM'] = [dis_codes_id_to_ave_dis_hpo_num[dis_codes_id] for dis_codes_id in dis_order]
			df_dict['PATIENT_HPO_AVE_NUM'] = [dis_codes_id_to_ave_pa_hpo_num[dis_codes_id] for dis_codes_id in dis_order]
			df = pd.DataFrame(df_dict, columns=columns).sort_values(model_names, ascending=(not reverse))
			df.to_excel(os.path.join(folder, '{}-{}-{}.xlsx'.format(self.eval_data, metric, data_name)), index=False)
			df.corr().to_excel(os.path.join(folder, '{}-{}-{}-corr.xlsx'.format(self.eval_data, metric, data_name)))


	def gen_case_result_xlsx(self, data_name, model_names, folder=None, sort=False):



		dataset = self.get_dataset(data_name)


		df_dict = {}
		columns = ['DATA_RANK', 'DISEASE_CODE', 'DISEASE_NAME', 'HPO_CODE', 'HPO_NAME'] + model_names
		for model_name in model_names:
			raw_results = self.load_raw_results(model_name, data_name)

			rank_list, _, _ = self.raw_results_to_importants(raw_results, dataset, 1, 20)


			df_dict[model_name] = rank_list
		df_dict['DATA_RANK'] = list(range(len(dataset)))
		df_dict['HPO_CODE'] = [hpo_list for hpo_list, dis_list in dataset]
		df_dict['DISEASE_CODE'] = [dis_list for hpo_list, dis_list in dataset]
		explainer = Explainer()
		df_dict['DISEASE_NAME'] = explainer.add_cns_info(df_dict['DISEASE_CODE'])
		df_dict['HPO_NAME'] = explainer.add_cns_info(df_dict['HPO_CODE'])
		df = pd.DataFrame(df_dict, columns=columns)
		if sort:
			df = df.sort_values(model_names)
		if folder is None:
			folder = self.get_case_result_folder()
		os.makedirs(folder, exist_ok=True)
		df.to_excel(os.path.join(folder, '{}.xlsx'.format(data_name)), index=False)


	def gen_save_all_metric_with_condition(self, model_name, data_name, new_data_name, choose_pa_func, raw_results=None):
		"""
		Args:
			model_name (str)
			choose_pa_func (func): args=((i, patient)), returns=True|False
		"""
		print('------------------------------\n%s, %s (SIZE=%d), calculating Metrics:' % (model_name, new_data_name, len(self.data[data_name])))
		if raw_results is None:
			raw_results = self.load_raw_results(model_name, data_name)
		metric_dict = self.cal_metric_dict_with_condition(self.data[data_name], raw_results, choose_pa_func)
		self.save_metric_dict(model_name, new_data_name, metric_dict)


	def draw_metric_bar(self, data_names, metric_names, model_names, fig_dir=None, dataset_order=None):

		if fig_dir is None:
			fig_dir = self.get_metric_bar_plot_fig_folder()
			os.makedirs(fig_dir, exist_ok=True)
		if dataset_order is None:
			dataset_order = data_names
		for metric_name in metric_names:
			assert metric_name in self.bar_metrics
			json_paths = [self.get_result_file_path(met, mod, dat, '.json') for met, mod, dat in itertools.product([metric_name], model_names, data_names)]
			fig_path = '{}/{}-{}.png'.format(fig_dir, self.eval_data, metric_name)
			draw_dodge_bar(json_paths, fig_path, x_col='Dataset', y_col=metric_name, class_col='Model', class_order=model_names, x_order=dataset_order)


	def draw_metric_quartile(self, data_names, metric_names, model_names, fig_dir=None, dataset_order=None):

		if fig_dir is None:
			fig_dir = self.get_metirc_quartile_plot_fig_folder()
			os.makedirs(fig_dir, exist_ok=True)
		if dataset_order is None:
			dataset_order = data_names
		for metric_name in metric_names:
			assert metric_name in self.quartile_metrics
			json_paths = [self.get_result_file_path(met, mod, dat, '.json') for met, mod, dat in itertools.product([metric_name], model_names, data_names)]
			fig_path = '{}/{}-{}.png'.format(fig_dir, self.eval_data, metric_name)
			draw_quartile_fig(json_paths, fig_path, x_col='Dataset', y_col=metric_name, class_col='Model', class_order=model_names, x_order=dataset_order)


	def draw_all_metric_given_models(self, model_names, fig_dir=None, dataset_order=None):
		fig_dir = fig_dir or self.RESULT_FIG_PATH
		self.draw_metric_bar(self.bar_metrics, model_names, fig_dir, dataset_order)
		self.draw_metric_quartile(self.quartile_metrics, model_names, fig_dir, dataset_order)


	def gen_result_df(self, metric_names, model_names, data_names, cal_model_rank=True, cal_dataset_mean=True, conf_level=None):
		def set_conf_int(data_name, model_name, metric_name, conf_int_dict, df):
			conf_int = conf_int_dict.get((data_name, model_name, metric_name), '')
			if not conf_int:
				return
			line_index = df[(df['Dataset'] == data_name) & (df['Model'] == model_name)].index
			if len(line_index) == 0:
				print('Not found:', data_name, model_name)
				return
			assert len(line_index) == 1
			line_index = line_index[0]
			if metric_name.find('RankMedian') > -1:
				df.loc[line_index, metric_name] = '{:.01f} ({:.01f}-{:.01f})'.format(
					df.loc[line_index, metric_name], conf_int[0], conf_int[1])
			else:
				df.loc[line_index, metric_name] = '{:.03f} ({:.03f}-{:.03f})'.format(
					df.loc[line_index, metric_name], conf_int[0], conf_int[1])
		def set_decimal_point(df):
			line_index = df[(df['Dataset'] != 'AVE_MODEL_RANK') & (df['Dataset'] != '')].index
			for metric_name in metric_names:
				if metric_name.find('RankMedian') > -1:
					df.loc[line_index, metric_name] = ['{:.1f}'.format(v) for v in df.loc[line_index, metric_name]]
				else:
					df.loc[line_index, metric_name] = ['{:.3f}'.format(v) for v in df.loc[line_index, metric_name]]

		columns = ['Dataset', 'Model'] + metric_names
		df = pd.DataFrame(columns=columns)
		conf_int_dict = {}
		empty_line_dict = dict({'Dataset': '', 'Model': ''}, **{metric_name: None for metric_name in metric_names})
		for data_name in data_names:
			for model_name in model_names:
				line_dict = {'Dataset':data_name, 'Model':model_name}
				for metric_name in metric_names:
					file_path = self.get_result_file_path(metric_name, model_name, data_name)
					if os.path.exists(file_path):
						result_dict = json.load(open(file_path))

						line_dict[metric_name] = result_dict[metric_name]
						conf_int_dict[(data_name, model_name, metric_name)] = result_dict.get(f'CONF_INT_{conf_level}', '')
					else:
						line_dict[metric_name] = None
				df = df.append(line_dict, ignore_index=True)
			df = df.append(empty_line_dict, ignore_index=True)
		if cal_dataset_mean:
			for model_name in model_names:
				line_df = df[df['Model'] == model_name].mean()
				line_df['Dataset'], line_df['Model'] = self.data_names_to_key(data_names), model_name
				df = df.append(line_df, ignore_index=True)
			data_name = self.data_names_to_key(data_names)
			for model_name, metric_name in itertools.product(model_names, metric_names):
				file_path = self.get_result_file_path(metric_name, model_name, data_name)
				if os.path.exists(file_path):
					conf_int_dict[(data_name, model_name, metric_name)] = json.load(open(file_path)).get(f'CONF_INT_{conf_level}', '')
			df = df.append(empty_line_dict, ignore_index=True)
		if cal_model_rank:
			df = df.append(self.cal_ave_model_ranking(metric_names, model_names, data_names), ignore_index=True)
		if conf_level is not None:
			for data_name, model_name, metric_name in itertools.product(data_names, model_names, metric_names):
				set_conf_int(data_name, model_name, metric_name, conf_int_dict, df)
			data_name = self.data_names_to_key(data_names)
			for model_name, metric_name in itertools.product(model_names, metric_names):
				set_conf_int(data_name, model_name, metric_name, conf_int_dict, df)
		else:
			set_decimal_point(df)
		return df


	def cal_ave_model_ranking(self, metric_names, model_names, data_names):
		"""
		Returns:
			pd.DataFrame: line_dict = {'Dataset': 'Ave ranking', 'Model': model_name, metric_name: score}
		"""
		metric_to_model_to_ave_rank = {}
		for metric_name in metric_names:
			model_to_rank_dicts = []
			for data_name in data_names:
				model_scores = []
				for model_name in model_names:
					file_path = self.get_result_file_path(metric_name, model_name, data_name)
					if not os.path.exists(file_path):
						raise RuntimeError('File not exists: {}'.format(file_path))
					result_dict = json.load(open(file_path))

					model_scores.append(result_dict[metric_name])



				model_ranks = pd.Series(model_scores).rank(ascending=(metric_name.find('RankMedian') > -1)).tolist()
				model_to_rank = {model_name: model_rank for model_name, model_rank in zip(model_names, model_ranks)}
				model_to_rank_dicts.append(model_to_rank)
				print(metric_name, data_name, model_to_rank)
			model_to_ave_rank = {model_name: np.average([d[model_name] for d in model_to_rank_dicts]) for model_name in model_names}
			metric_to_model_to_ave_rank[metric_name] = model_to_ave_rank
		columns = ['Dataset', 'Model'] + metric_names
		df = pd.DataFrame(columns=columns)
		for model_name in model_names:
			line_dict = {'Dataset': 'AVE_MODEL_RANK', 'Model': model_name}
			line_dict.update({metric_name: '{:.1f}'.format(metric_to_model_to_ave_rank[metric_name][model_name]) for metric_name in metric_names})
			df = df.append(line_dict, ignore_index=True)
		return df


	def gen_ave_source_ranking_excel(self, metric_name, model_names, data_names, source_marks, excel_path=None, use_mark=True):
		"""
		Args:
			sources_lists (list): e.g.['OMIM_ORPHA', 'CCRD']
		Returns:
			pd.DataFrame: index=model_names; columns=source_marks
		"""
		def get_metric(metric_name, model_name, data_name, source_mark, use_mark):
			if use_mark:
				marked_model_name = self.get_model_name_with_mark(model_name, source_mark)
				file_path = self.get_result_file_path(metric_name, marked_model_name, data_name)
			else:
				tmp_result_path = self.RESULT_PATH
				self.RESULT_PATH = os.path.join(RESULT_PATH, source_mark)
				file_path = self.get_result_file_path(metric_name, model_name, data_name)
				self.RESULT_PATH = tmp_result_path
			if not os.path.exists(file_path):
				raise RuntimeError('File not exists: {}'.format(file_path))
			result_dict = json.load(open(file_path))
			return result_dict[metric_name]

		model_to_source_to_ave_rank = {}
		for model_name in model_names:
			source2rank_list = []
			for data_name in data_names:
				source_scores = []
				for source_mark in source_marks:
					source_scores.append(get_metric(metric_name, model_name, data_name, source_mark, use_mark))
				source_ranks = pd.Series(source_scores).rank(ascending=(metric_name.find('RankMedian') > -1)).tolist()
				print(model_name, data_name, source_ranks)
				source2rank = {source_mark:source_rank for source_mark, source_rank in zip(source_marks, source_ranks)}
				source2rank_list.append(source2rank)
			source_to_ave_rank = {source_mark: np.average([d[source_mark] for d in source2rank_list]) for source_mark in source_marks}
			model_to_source_to_ave_rank[model_name] = source_to_ave_rank
		df = pd.DataFrame(columns=['Model'] + source_marks)
		for model_name in model_names:
			line_dict = {'Model': model_name}
			line_dict.update({source_mark: model_to_source_to_ave_rank[model_name][source_mark] for source_mark in source_marks})
			df = df.append(line_dict, ignore_index=True)
		excel_path = excel_path or (self.get_table_result_folder() + '/{}-source-ranking.xlsx'.format(self.eval_data))
		df.to_excel(excel_path, index=False)


	def gen_result_xlsx(self, metric_names, model_names, data_names, filepath=None,
			cal_model_rank=True, cal_dataset_mean=True, conf_level=None):
		if filepath is None:
			filepath = self.get_table_result_folder() + '/{}-result.xlsx'.format(self.eval_data)
		df = self.gen_result_df(metric_names, model_names, data_names, cal_model_rank=cal_model_rank,
			cal_dataset_mean=cal_dataset_mean, conf_level=conf_level)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		df.to_excel(filepath, index=False)


	def gen_result_csv(self, metric_names, model_names, data_names, filepath=None):
		if filepath is None:
			filepath = self.get_table_result_folder() + '/{}-result.csv'.format(self.eval_data)
		df = self.gen_result_df(metric_names, model_names, data_names)
		df.to_csv(filepath, index=False)


	def delete_result(self, model_name):
		for metric_name in self.all_metrics:
			shutil.rmtree(self.get_metric_model_path(metric_name, model_name), ignore_errors=True)


	def change_data_name(self, old_data_name, new_data_name, path=None, exclude=set()):
		if path is None:
			path = self.get_eval_data_path()
		self.change_name(
			path, change_folder=False, change_file=True, change_key_value=True,
			old_file_name=old_data_name+'.json', new_file_name=new_data_name+'.json',
			old_key='Dataset', old_key_value=old_data_name, new_key='Dataset', new_key_value=new_data_name, exclude=exclude
		)


	def change_model_name(self, oldModelName, new_model_name, path=None, exclude=set()):
		if path is None:
			path = self.get_eval_data_path()
		self.change_name(
			path, change_folder=True, change_file=False, change_key_value=True,
			old_folder_name=oldModelName, new_folder_name=new_model_name,
			old_key='Model', old_key_value=oldModelName, new_key='Model', new_key_value=new_model_name, exclude=exclude
		)


	def change_metric_name(self, old_metric_name, new_metric_name, path=None, exclude=set()):
		if path is None:
			path = self.get_eval_data_path()
		self.change_name(
			path, change_folder=True, change_file=False, change_key_value=True,
			old_folder_name=old_metric_name, new_folder_name=new_metric_name,
			old_key=old_metric_name, old_key_value=None, new_key=new_metric_name, new_key_value=None, exclude=exclude
		)


	def change_name(self, path, change_folder=False, change_file=False, change_key_value=False, old_folder_name=None, new_folder_name=None,
			old_file_name=None, new_file_name=None, old_key=None, old_key_value=None, new_key=None, new_key_value=None, exclude=set()):
		if path in exclude:
			return
		if os.path.isfile(path):
			folder, file_name = os.path.split(path)
			fprefix, fpostfix = os.path.splitext(file_name)
			if change_key_value and fpostfix == '.json':
				result_dict = json.load(open(path))
				if old_key != new_key and old_key in result_dict:
					print('Change Key: {}'.format(path))
					result_dict[new_key] = result_dict[old_key]
					del result_dict[old_key]
				if new_key in result_dict and result_dict[new_key] == old_key_value:
					print('Change Key Value: {}'.format(path))
					result_dict[new_key] = new_key_value
				json.dump(result_dict, open(path, 'w'), indent=2)
			if change_file and file_name == old_file_name:
				print('Rename File: {}'.format(path))
				os.rename(path, os.path.join(folder, new_file_name))
		else:
			for file_name in os.listdir(path):
				self.change_name(
					os.path.join(path, file_name), change_folder, change_file, change_key_value, old_folder_name, new_folder_name,
					old_file_name, new_file_name, old_key, old_key_value, new_key, new_key_value, exclude
				)
			upPath, folder_name = os.path.split(path)
			if change_folder and folder_name == old_folder_name:
				assert old_folder_name != new_folder_name
				print('Change File Folder: {0}'.format(path))
				new_path = os.path.join(upPath, new_folder_name)
				shutil.rmtree(new_path, ignore_errors=True)
				shutil.move(path, new_path)


	def rank_score_ensemble(self, model_names, data_names, ensemble_name, model_weight=None,
			metric_set=None, logger=None, save=False, cpu_use=8, combine_method='ave', hpo_reader=None, rd_decompose=False):
		metric_set = metric_set or self.all_metrics
		model = RankScoreModel(hpo_reader=hpo_reader or self.get_hpo_reader())
		for data_name in data_names:
			models_raw_results = [self.load_raw_results(model_name, data_name) for model_name in model_names]

			if logger is not None:
				logger.info(
					'\n------------------------------\n{}, {} (SIZE={}), calculating Metrics:'.format(
						ensemble_name, data_name, len(models_raw_results[0])))
			raw_results = model.combine_many_raw_results(models_raw_results, model_weight, cpu_use=cpu_use, combine_method=combine_method)
			if save:
				self.save_raw_results(raw_results, ensemble_name, data_name)

			metric_dict = self.get_performance(raw_results, self.data[data_name], metric_set, logger, cpu_use=cpu_use, rd_decompose=rd_decompose)
			self.save_metric_dict(ensemble_name, data_name, metric_dict)


	def consistency_ensemble(self, tgt_model_name, all_model_names, data_names, ensemble_name,
			topk, threshold, metric_set=None, logger=None, save=False, cpu_use=8, hpo_reader=None):
		metric_set = metric_set or self.all_metrics
		model = ConsistencyRateModel(hpo_reader=hpo_reader or self.get_hpo_reader())
		for data_name in data_names:
			raw_results = self.load_raw_results(tgt_model_name, data_name)
			models_raw_results = [self.load_raw_results(model_name, data_name) for model_name in all_model_names]
			if logger is not None:
				logger.info(
					'\n------------------------------\n{}, {} (SIZE={}), calculating Metrics:'.format(
						ensemble_name, data_name, len(models_raw_results[0])))
			raw_results = model.rerank_raw_results(raw_results, models_raw_results, topk, threshold, cpu_use=cpu_use)
			if save:
				self.save_raw_results(raw_results, ensemble_name, data_name)
			metric_dict = self.get_performance(raw_results, self.data[data_name], metric_set, logger, cpu_use=cpu_use)
			self.save_metric_dict(ensemble_name, data_name, metric_dict)


	def get_p_value(self, pred_ranks1, pred_ranks2, alternative, metric, r_object):
		if metric == 'Mic.RankMedian':
			return py_wilcox(pred_ranks1, pred_ranks2, alternative=alternative, paired=False, robjects=r_object)['p_value']

		elif metric.startswith('Mic.Recall'):
			top_n = int(metric.split('.').pop())
			top_n_hits1 = np.array(self.ranks_to_top_n_hits(pred_ranks1, top_n), dtype=np.bool)
			top_n_hits2 = np.array(self.ranks_to_top_n_hits(pred_ranks2, top_n), dtype=np.bool)
			mcnemar_table = [
				[np.sum(top_n_hits1 & top_n_hits2), np.sum(top_n_hits1 & ~top_n_hits2)],
				[np.sum(~top_n_hits1 & top_n_hits2), np.sum(~top_n_hits1 & ~top_n_hits2)]
			]
			return cal_mcnemar_p_value(mcnemar_table)
		else:
			raise RuntimeError('Unknown compare: {}'.format(metric))


	def cal_pairwise_pvalue_df(self, model_names1, model_names2, data_name, alternative='two.sided', metric='Mic.RankMedian', cpu_use=12):
		"""
		Returns:
			pd.DataFrame: shape=(len(model_names1), len(model_names2))
		"""
		def get_and_check_pred_ranks(model_name):
			if model_name not in name_to_pred_ranks:
				name_to_pred_ranks[model_name] = self.get_pred_ranks(model_name, data_name)
			return name_to_pred_ranks[model_name]


		r_object, importr = import_R()
		name_to_pred_ranks = {}
		for model_name in model_names2:
			get_and_check_pred_ranks(model_name)
		pvalues = []
		for model_name in tqdm(model_names1):
			row = []
			pred_ranks1 = get_and_check_pred_ranks(model_name)
			for model_name2 in model_names2:
				if model_name == model_name2:
					row.append(None)
				else:
					pred_ranks2 = get_and_check_pred_ranks(model_name2)
					row.append(self.get_p_value(pred_ranks1, pred_ranks2, alternative, metric, r_object))

			pvalues.append(row)
		return pd.DataFrame(np.vstack(pvalues), index=model_names1, columns=model_names2)


	def get_multi_mean_diff_int_for_two_multi_ranks(self, multi_pred_ranks1, multi_pred_ranks2, metric, conf_level=0.95):
		if metric == 'Mic.RankMedian':
			diff, interval = self.cal_multi_mean_diff_conf_interval(multi_pred_ranks1, multi_pred_ranks2,
				metric_type='mean_median_diff', conf_level=conf_level)
		elif metric.startswith('Mic.Recall'):
			top_n = int(metric.split('.').pop())
			multi_top_n_hits1 = [self.ranks_to_top_n_hits(pred_ranks1, top_n) for pred_ranks1 in multi_pred_ranks1]
			multi_top_n_hits2 = [self.ranks_to_top_n_hits(pred_ranks2, top_n) for pred_ranks2 in multi_pred_ranks2]
			diff, interval = self.cal_multi_mean_diff_conf_interval(multi_top_n_hits1, multi_top_n_hits2,
				metric_type='mean_mean_diff', conf_level=conf_level)
		else:
			raise RuntimeError('Unknown compare: {}'.format(metric))
		return diff, interval


	def get_multi_mean_diff_int_for_two_multi_ranks_wrapper(self, paras):
		return self.get_multi_mean_diff_int_for_two_multi_ranks(*paras)


	def get_diff_conf_int_for_two_ranks(self, pred_ranks1, pred_ranks2, metric, conf_level=0.95):
		if metric == 'Mic.RankMedian':
			diff = np.median(pred_ranks1) - np.median(pred_ranks2)
			interval = self.cal_diff_conf_interval(pred_ranks1, pred_ranks2, metric_type='median_diff', conf_level=conf_level)
		elif metric.startswith('Mic.Recall'):
			top_n = int(metric.split('.').pop())
			top_n_hits1 = np.array(self.ranks_to_top_n_hits(pred_ranks1, top_n), dtype=np.bool)
			top_n_hits2 = np.array(self.ranks_to_top_n_hits(pred_ranks2, top_n), dtype=np.bool)
			diff = np.mean(top_n_hits1) - np.mean(top_n_hits2)
			interval = self.cal_diff_conf_interval(top_n_hits1, top_n_hits2, metric_type='mean_diff', conf_level=conf_level)
		else:
			raise RuntimeError('Unknown compare: {}'.format(metric))
		return diff, interval


	def get_diff_conf_int_for_two_ranks_wrapper(self, paras):
		return self.get_diff_conf_int_for_two_ranks(*paras)


	def cal_pairwise_diff_conf_int_df(self, model_names1, model_names2, data_name, conf_level=0.95, metric='Mic.RankMedian', cpu_use=12, rd_decompose=False):
		def get_iterator():
			for model_name1, model_name2 in itertools.product(model_names1, model_names2):
				yield name_to_pred_ranks[model_name1], name_to_pred_ranks[model_name2], metric, conf_level

		name_to_pred_ranks = {}

		df = pd.DataFrame(index=model_names1, columns=model_names2)
		with Pool(cpu_use) as pool:
			unq_model_names = set(model_names1 + model_names2)
			paras = [(model_name, data_name, rd_decompose) for model_name in unq_model_names]
			for pred_ranks, model_name, data_name in pool.imap(self.get_pred_ranks_wrapper, paras):
				name_to_pred_ranks[model_name] = pred_ranks
			res = pool.map(self.get_diff_conf_int_for_two_ranks_wrapper, get_iterator())
			for i, (model_name1, model_name2) in enumerate(itertools.product(model_names1, model_names2)):
				diff, interval = res[i]
				if metric.find('RankMedian') > -1:
					df.loc[model_name1, model_name2] = '{:.01f} ({:.01f}, {:.01f})'.format(diff, interval[0], interval[1])
				else:
					df.loc[model_name1, model_name2] = '{:.03f} ({:.03f}, {:.03f})'.format(diff, interval[0], interval[1])
		return df


	def gen_pairwise_multi_mean_diff_conf_int_excel(self, model_names1, model_names2, data_names,
			conf_level=0.95, metric='Mic.RankMedian', cpu_use=12, rd_decompose=False):
		def get_iterator():
			for model_name1, model_name2 in itertools.product(model_names1, model_names2):
				multi_pred_ranks1 = [name_to_pred_ranks[(model_name1, data_name)] for data_name in data_names]
				multi_pred_ranks2 = [name_to_pred_ranks[(model_name2, data_name)] for data_name in data_names]
				yield multi_pred_ranks1, multi_pred_ranks2, metric, conf_level

		name_to_pred_ranks = {}
		df = pd.DataFrame(index=model_names1, columns=model_names2)

		with Pool(cpu_use) as pool:
			unq_model_names = set(model_names1 + model_names2)
			paras = [(model_name, data_name, rd_decompose) for model_name, data_name in itertools.product(unq_model_names, data_names)]
			for pred_ranks, model_name, data_name in pool.imap(self.get_pred_ranks_wrapper, paras):
				name_to_pred_ranks[(model_name, data_name)] = pred_ranks
			res = pool.map(self.get_multi_mean_diff_int_for_two_multi_ranks_wrapper, get_iterator())
			for i, (model_name1, model_name2) in enumerate(itertools.product(model_names1, model_names2)):
				diff, interval = res[i]
				if metric.find('RankMedian'):
					df.loc[model_name1, model_name2] = '{:.01f} ({:.01f}, {:.01f})'.format(diff, interval[0], interval[1])
				else:
					df.loc[model_name1, model_name2] = '{:.03f} ({:.03f}, {:.03f})'.format(diff, interval[0], interval[1])
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}_multi_mean_diff_int_{conf_level}.xlsx')
		df.to_excel(xlsx_path)


	def gen_pairwise_diff_int_excel(self, model_names1, model_names2, data_names,
			conf_level=0.95, metric='Mic.RankMedian', cpu_use=12, sheet_order_by_swap=False):
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}_diff_int_{conf_level}.xlsx')
		writer = pd.ExcelWriter(xlsx_path)
		df_dict = {data_name: self.cal_pairwise_diff_conf_int_df(
			model_names1, model_names2, data_name, conf_level=conf_level, metric=metric) for data_name in data_names}
		if sheet_order_by_swap:
			for model_name2 in model_names2:
				sub_df = pd.DataFrame(index=model_names2, columns=data_names)
				for model_name1, data_name in itertools.product(model_names1, data_names):
					sub_df.loc[model_name1, data_name] = df_dict[data_name].loc[model_name1, model_name2]
				sub_df.to_excel(writer, sheet_name=self.process_sheet_name(model_name2))
		else:
			for model_name1 in model_names1:
				sub_df = pd.DataFrame(index=model_names1, columns=data_names)
				for model_name2, data_name in itertools.product(model_names2, data_names):
					sub_df.loc[model_name2, data_name] = df_dict[data_name].loc[model_name1, model_name2]
				sub_df.to_excel(writer, sheet_name=self.process_sheet_name(model_name1))


	def gen_dataset_diff_int_excel(self, model_names, data_name_pairs, conf_level=0.95, metric='Mic.RankMedian', cpu_use=12, rd_decompose=False):
		def get_iterator():
			for model_name, (dname1, dname2) in itertools.product(model_names, data_name_pairs):
				yield name_to_pred_ranks[(model_name, dname1)], name_to_pred_ranks[(model_name, dname2)], metric, conf_level
		name_to_pred_ranks = {}
		df = pd.DataFrame(index=model_names)
		with Pool(cpu_use) as pool:
			unq_model_data = set()
			for model_name, (dname1, dname2) in itertools.product(model_names, data_name_pairs):
				unq_model_data.add((model_name, dname1, rd_decompose))
				unq_model_data.add((model_name, dname2, rd_decompose))
			unq_model_data = list(unq_model_data)
			for pred_ranks, model_name, data_name in pool.imap(self.get_pred_ranks_wrapper, unq_model_data):
				name_to_pred_ranks[(model_name, data_name)] = pred_ranks
			res = pool.map(self.get_diff_conf_int_for_two_ranks_wrapper, get_iterator())
			for i, (model_name, (dname1, dname2)) in enumerate(itertools.product(model_names, data_name_pairs)):
				diff, interval = res[i]
				if metric.find('RankMedian') > -1:
					df.loc[model_name, f'{dname1}-vs-{dname2}'] = '{:.01f} ({:.01f}, {:.01f})'.format(diff, interval[0], interval[1])
				else:
					df.loc[model_name, f'{dname1}-vs-{dname2}'] = '{:.03f} ({:.03f}, {:.03f})'.format(diff, interval[0], interval[1])
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}_dataset_diff_int_{conf_level}.xlsx')
		df.to_excel(xlsx_path)


	def gen_dataset_pavalue_excel(self, model_names, data_name_pairs, alternative='two.sided',
			metric='Mic.RankMedian', multi_test_cor=None, cpu_use=12, rd_decompose=False):
		r_object, importr = import_R()
		name_to_pred_ranks = {}
		df = pd.DataFrame(columns=model_names)
		with Pool(cpu_use) as pool:
			unq_model_data = set()
			for model_name, (dname1, dname2) in itertools.product(model_names, data_name_pairs):
				unq_model_data.add((model_name, dname1, rd_decompose))
				unq_model_data.add((model_name, dname2, rd_decompose))
			unq_model_data = list(unq_model_data)
			for pred_ranks, model_name, data_name in pool.imap(self.get_pred_ranks_wrapper, unq_model_data):
				name_to_pred_ranks[(data_name, model_name)] = pred_ranks

			for (dname1, dname2) in data_name_pairs:
				index_name = f'{dname1}-vs-{dname2}'
				for model_name in model_names:
					pvalue = self.get_p_value(
						name_to_pred_ranks[(dname1, model_name)], name_to_pred_ranks[(dname2, model_name)],
						alternative, metric, r_object)
					df.loc[index_name, model_name] = pvalue
				if multi_test_cor is not None:
					df.loc[index_name] = pvalue_correct(df.loc[index_name].values, multi_test_cor)
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}-pvalue-{alternative}-multi_cor{multi_test_cor}-dataset_compare.xlsx')
		df.to_excel(xlsx_path)


	def gen_source_compare_pvalue_excel(self, model_names, data_names, source_mark_pairs,
			alternative='two.sided', metric='Mic.RankMedian', multi_test_cor=None):
		""""""
		r_object, importr = import_R()
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}-pvalue-{alternative}-multi_cor{multi_test_cor}-source_compare.xlsx')
		writer = pd.ExcelWriter(xlsx_path)
		for source_mark1, source_mark2 in source_mark_pairs:
			sheet_name = self.process_sheet_name(f'{source_mark1}-vs-{source_mark2}')
			df = pd.DataFrame(index=model_names, columns=data_names)
			for data_name in data_names:
				for model_name in model_names:
					model_name1 = self.get_model_name_with_mark(model_name, source_mark1)
					model_name2 = self.get_model_name_with_mark(model_name, source_mark2)
					pred_ranks1, pred_ranks2 = self.get_pred_ranks(model_name1, data_name), self.get_pred_ranks(model_name2, data_name)
					pvalue = self.get_p_value(pred_ranks1, pred_ranks2, alternative, metric, r_object)
					df.loc[model_name, data_name] = pvalue
				if multi_test_cor is not None:
					df[data_name] = pvalue_correct(df[data_name].values, multi_test_cor)
			df.to_excel(writer, sheet_name=sheet_name)


	def get_pred_ranks(self, model_name, data_name, rd_decompose=False):
		if model_name == 'Phenomizer':
			patients = self.get_phenomizer_test_data(data_name)
		else:
			if data_name not in self.data:
				self.load_test_data([data_name])
			patients = self.data[data_name]
		raw_results = self.load_raw_results(model_name, data_name)
		rank_list, _, _ = self.raw_results_to_importants(raw_results, patients, 1, chunk_size=20, rd_decompose=rd_decompose)
		return rank_list


	def get_pred_ranks_wrapper(self, paras):
		if len(paras) == 2:
			model_name, data_name = paras
			return self.get_pred_ranks(model_name, data_name), model_name, data_name
		elif len(paras) == 3:
			model_name, data_name, rd_decompose = paras
			return self.get_pred_ranks(model_name, data_name, rd_decompose), model_name, data_name
		assert False


	def get_source_compare_hpo_reader(self, source_mark):
		if self.use_rd_code:
			return HPOIntegratedDatasetReader(keep_dnames=source_mark[len('INTEGRATE_'):].split('_'))
		else:
			return HPOFilterDatasetReader(keep_dnames=source_mark.split('_'))


	def cal_source_compare_multi_mean_diff_wrapper(self, paras):
		model_name1, model_name2, conf_level, model_name, data_names, metric, source_mark1, source_mark2, use_mark = paras
		if use_mark:
			multi_pred_ranks1 = [self.get_pred_ranks(model_name1, data_name) for data_name in data_names]
			multi_pred_ranks2 = [self.get_pred_ranks(model_name2, data_name) for data_name in data_names]
		else:
			for data_name in data_names:
				if data_name not in self.data:
					self.load_test_data([data_name])
			tmp_result_path = self.RESULT_PATH
			tmp_hpo_reader = self.hpo_reader
			self.RESULT_PATH = os.path.join(RESULT_PATH, source_mark1)
			self.hpo_reader = self.get_source_compare_hpo_reader(source_mark1)
			multi_pred_ranks1 = [self.get_pred_ranks(model_name, data_name) for data_name in data_names]
			self.RESULT_PATH = os.path.join(RESULT_PATH, source_mark2)
			self.hpo_reader = self.get_source_compare_hpo_reader(source_mark2)
			multi_pred_ranks2 = [self.get_pred_ranks(model_name, data_name) for data_name in data_names]
			self.RESULT_PATH = tmp_result_path
			self.hpo_reader = tmp_hpo_reader
		diff, interval = self.get_multi_mean_diff_int_for_two_multi_ranks(multi_pred_ranks1, multi_pred_ranks2, metric, conf_level=conf_level)
		return diff, interval, model_name, '-'.join(data_names)


	def cal_source_compare_diff_wrapper(self, paras):
		model_name1, model_name2, conf_level, model_name, data_name, metric, source_mark1, source_mark2, use_mark = paras
		if use_mark:
			pred_ranks1, pred_ranks2 = self.get_pred_ranks(model_name1, data_name), self.get_pred_ranks(model_name2, data_name)
		else:
			if data_name not in self.data:
				self.load_test_data([data_name])
			tmp_result_path = self.RESULT_PATH
			tmp_hpo_reader = self.hpo_reader
			self.RESULT_PATH = os.path.join(RESULT_PATH, source_mark1)
			self.hpo_reader = self.get_source_compare_hpo_reader(source_mark1)
			pred_ranks1 = self.get_pred_ranks(model_name, data_name)
			self.RESULT_PATH = os.path.join(RESULT_PATH, source_mark2)
			self.hpo_reader = self.get_source_compare_hpo_reader(source_mark2)
			pred_ranks2 = self.get_pred_ranks(model_name, data_name)
			self.RESULT_PATH = tmp_result_path
			self.hpo_reader = tmp_hpo_reader
		diff, interval = self.get_diff_conf_int_for_two_ranks(pred_ranks1, pred_ranks2, metric, conf_level=conf_level)
		return diff, interval, model_name, data_name


	def gen_source_compare_diff_int_excel(self, model_names, data_names, source_mark_pairs,
			metric='Mic.RankMedian', conf_level=0.95, cpu_use=12, multi_mean=False, use_mark=True):
		def get_iterator(model_names, data_names, source_mark1, source_mark2):
			for data_name in data_names:
				print(f'Compare {source_mark1} and {source_mark2} for {data_name}')
				for model_name in model_names:
					model_name1 = self.get_model_name_with_mark(model_name, source_mark1)
					model_name2 = self.get_model_name_with_mark(model_name, source_mark2)
					yield model_name1, model_name2, conf_level, model_name, data_name, metric, source_mark1, source_mark2, use_mark
		def get_multi_mean_iterator(model_names, data_names, source_mark1, source_mark2):
			print(f'Compare {source_mark1} and {source_mark2} for {data_name}')
			for model_name in model_names:
				model_name1 = self.get_model_name_with_mark(model_name, source_mark1)
				model_name2 = self.get_model_name_with_mark(model_name, source_mark2)
				yield model_name1, model_name2, conf_level, model_name, data_names, metric, source_mark1, source_mark2, use_mark
		def get_short_integrate_name(source_mark):
			source_names = source_mark.split('_')
			return '_'.join([s[:2] for s in source_names])

		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}_diff_{conf_level}-source_compare.xlsx')
		writer = pd.ExcelWriter(xlsx_path)
		for source_mark1, source_mark2 in source_mark_pairs:
			sheet_name = self.process_sheet_name(f'{get_short_integrate_name(source_mark1)}-vs-{get_short_integrate_name(source_mark2)}')
			df = pd.DataFrame(index=model_names, columns=data_names)
			with Pool(cpu_use) as pool:
				for diff, interval, model_name, data_name in pool.map(
						self.cal_source_compare_diff_wrapper,
						get_iterator(model_names, data_names, source_mark1, source_mark2)):
					if metric.find('RankMedian') > -1:
						df.loc[model_name, data_name] = '{:.01f} ({:.01f}, {:.01f})'.format(
							diff, interval[0], interval[1])
					else:
						df.loc[model_name, data_name] = '{:.03f} ({:.03f}, {:.03f})'.format(
							diff, interval[0], interval[1])
				if multi_mean:
					for diff, interval, model_name, data_name in pool.map(
						self.cal_source_compare_multi_mean_diff_wrapper,
						get_multi_mean_iterator(model_names, data_names, source_mark1, source_mark2)):
						if metric.find('RankMedian') > -1:
							df.loc[model_name, data_name] = '{:.01f} ({:.01f}, {:.01f})'.format(
								diff, interval[0], interval[1])
						else:
							df.loc[model_name, data_name] = '{:.03f} ({:.03f}, {:.03f})'.format(
								diff, interval[0], interval[1])
			df.to_excel(writer, sheet_name=sheet_name)


	def get_pvalue_display(self, p):
		if p >= 0.01:
			return '{:.2f}'.format(p)
		elif p >= 0.001:
			return '{:.3f}'.format(p)
		else:
			return '{:.2e}'.format(p)


	def gen_pvalue_table(self, model_names1, model_names2, data_names, alternative='two.sided',
			multi_test_cor=None, metric='Mic.RankMedian'):
		xlsx_path = os.path.join(self.get_table_result_folder(),
			f'{self.eval_data}-{metric}-pvalue-{alternative}-multi_cor{multi_test_cor}.xlsx')
		writer = pd.ExcelWriter(xlsx_path)
		df_dict = {data_name: self.cal_pairwise_pvalue_df(model_names1, model_names2, data_name, alternative, metric=metric) for data_name in data_names}
		for model_name1 in model_names1:
			sub_df = pd.DataFrame(index=model_names2, columns=data_names)
			for model_name2, data_name in itertools.product(model_names2, data_names):
				sub_df.loc[model_name2, data_name] = df_dict[data_name].loc[model_name1, model_name2]
			if multi_test_cor is not None:
				for index, row_se in sub_df.iterrows():
					pvals = row_se.values
					if pvals[0] is not None:
						pvals_cor = pvalue_correct(pvals, multi_test_cor)
						sub_df.loc[index] = ['{} ({})'.format(
							self.get_pvalue_display(p_cor), self.get_pvalue_display(p)) for p, p_cor in zip(pvals, pvals_cor)]
			else:
				sub_df = sub_df.apply(self.get_pvalue_display)
			sub_df.to_excel(writer, sheet_name=self.process_sheet_name(model_name1))


	def gen_pvalue_table_sheet_by_data(self, model_names1, model_names2, data_names, alternative='two.sided', metric='Mic.RankMedian'):
		xlsx_path = os.path.join(self.get_table_result_folder(), f'{self.eval_data}-pvalue.xlsx')
		writer = pd.ExcelWriter(xlsx_path)
		for data_name in data_names:
			print('Calculating p-value for {} ...'.format(data_name))
			df = self.cal_pairwise_pvalue_df(model_names1, model_names2, data_name, alternative, metric=metric)
			df.to_excel(writer, sheet_name=self.process_sheet_name(data_name))
		writer.close()


	def cal_level_performance(self, model_name, data_name, levels, metric_set, save=True):
		"""
		Args:
			levels (list): contains DISORDER_GROUP_LEAF_LEVEL | DISORDER_GROUP_LEAF_LEVEL | DISORDER_LEVEL | DISORDER_SUBTYPE_LEVEL
		Returns:
			dict: {metric_name: value}
		"""
		raw_results = self.load_raw_results(model_name, data_name)
		self.load_test_data([data_name])
		patients = self.data[data_name]
		print('\n------------------------------\n{}, {} (SIZE={}), calculating Metrics (Levels: {}):'.format(
			model_name, data_name, len(patients), levels))

		rd_reader = RDReader()
		rd_dict = rd_reader.get_rd_dict()
		if DISORDER_GROUP_LEAF_LEVEL in levels:
			rd_reader.set_level_leaf_codes(rd_dict, DISORDER_GROUP_LEVEL, DISORDER_GROUP_LEAF_LEVEL)
		rd2ances = {rd: get_all_ancestors(rd, rd_dict) for rd in rd_dict}
		dis2rd = rd_reader.get_source_to_rd()
		rd2level = rd_reader.get_rd_to_level()
		rank_list, top_n_hit_lists = [], []

		for result, (hpos, diag_dis_list) in tqdm(zip(raw_results, patients), total=len(patients)):
			dis_list, score_list = zip(*result)
			rank = self.get_level_rank(dis_list, diag_dis_list, levels, rd2ances, dis2rd, rd2level)
			top_n_hit = self.rank_to_top_n_hit(rank)
			rank_list.append(rank); top_n_hit_lists.append(top_n_hit)

		rank_array, top_n_hit_matrix = np.array(rank_list), np.array(top_n_hit_lists)
		ret_dict = {}
		rank_quatile = self.cal_quartile(rank_array)
		print('[minimum, Q1, median, Q3, maximum] = [%.2f, %.2f, %.2f, %.2f, %.2f]' % tuple(rank_quatile))
		if 'Mic.RankMedian' in metric_set:
				ret_dict['Mic.RankMedian'] = rank_quatile[2]
		mic_recall_n = top_n_hit_matrix.mean(axis=0)    # shape=[len(top_n_list)]
		for i in range(len(self.top_n_list)):
			k = 'Mic.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = mic_recall_n[i]
		dis_codes_to_idx_list = {}
		for i, pa in enumerate(patients):
			dict_list_add(self.dis_codes_to_id(pa[1]), i, dis_codes_to_idx_list)
		if 'Mac.RankMedian' in metric_set:
			ret_dict['Mac.RankMedian'] = np.median([self.cal_quartile(rank_array[idx_list])[2] for idx_list in dis_codes_to_idx_list.values()])
		mac_recall_n = np.array([top_n_hit_matrix[idx_list].mean(axis=0) for idx_list in dis_codes_to_idx_list.values()]).mean(axis=0)
		for i in range(len(self.top_n_list)):
			k = 'Mac.Recall.%d' % (self.top_n_list[i])
			if k in metric_set:
				ret_dict[k] = mac_recall_n[i]
		for ret_item in ret_dict.items():
			print('%s: %s' % ret_item)
		if 'Rank' in metric_set:
			ret_dict['Rank'] = rank_list
		mark = '_'.join(sorted(levels))
		self.save_metric_dict(model_name, data_name, {f'{metric_name}_{mark}': value for metric_name, value in ret_dict.items()})


	def get_level_rank(self, dis_list, tgt_dis_list, levels, rd2ances, dis2rd, rd2level):
		tgt_rd_ances = set()
		for tgt_dis in tgt_dis_list:
			tgt_rd_ances.update(rd2ances[dis2rd[tgt_dis]])
		tgt_rd_ances = {rd for rd in tgt_rd_ances if rd2level[rd] in levels}
		tgt_rd_ances.update([dis2rd[tgt_dis] for tgt_dis in tgt_dis_list])
		tgt_dis_set = set(tgt_dis_list)

		for i, dis in enumerate(dis_list, 1):
			if dis in tgt_dis_set:
				return i
			cand_rd_ances = rd2ances[dis2rd[dis]]
			if len(tgt_rd_ances & cand_rd_ances) > 0:
				return i
		assert False


	def rank_to_top_n_hit(self, rank):
		"""rank \in [1, inf)
		Returns:
			list: length = len(self.top_n_list)
		"""
		top_n_hit = [0]*len(self.top_n_list) if rank <= 0 else [1 if rank <= top_n else 0 for top_n in self.top_n_list]    # 0-1 vec, shape=[len(top_n_list), ]
		return top_n_hit


	def ranks_to_top_n_hits(self, ranks, top_n):
		"""ranks starts from 1
		Returns:
			list: length = len(ranks)
		"""
		return [1 if rank <= top_n else 0  for rank in ranks]


	def cal_diff_conf_interval(self, x, y, metric_type, conf_level=0.95):
		if metric_type == 'mean_diff':
			return cal_boot_conf_int_for_multi_x([x, y], lambda a_list: np.mean(a_list[0]) - np.median(a_list[1]), conf_level=conf_level)
		elif metric_type == 'median_diff':
			return cal_boot_conf_int_for_multi_x([x, y], lambda a_list: np.median(a_list[0]) - np.median(a_list[1]), conf_level=conf_level)
		else:
			raise RuntimeError('Unknown metric_type', metric_type)


	def cal_multi_mean_diff_conf_interval(self, xlist, ylist, metric_type, conf_level=0.95):
		def cal_mean_median(a_list):
			return np.mean([np.median(a) for a in a_list])
		def cal_mean_mean(a_list):
			return np.mean([np.mean(a) for a in a_list])
		xylist = xlist + ylist
		x_list_num = len(xlist)
		if metric_type == 'mean_median_diff':
			diff = cal_mean_median(xlist) - cal_mean_median(ylist)
			return diff, cal_boot_conf_int_for_multi_x(
				xylist, lambda a_list:cal_mean_median(a_list[:x_list_num]) - cal_mean_median(a_list[x_list_num:]),
				conf_level=conf_level)
		elif metric_type == 'mean_mean_diff':
			diff = cal_mean_mean(xlist) - cal_mean_mean(ylist)
			return diff, cal_boot_conf_int_for_multi_x(
				xylist, lambda a_list: cal_mean_mean(a_list[:x_list_num]) - cal_mean_mean(a_list[x_list_num:]),
				conf_level=conf_level)
		else:
			raise RuntimeError('Unknown metric_type: {}'.format(metric_type))


	def cal_conf_interval(self, x, metric_type, id_lists=None, conf_level=0.95):
		"""
		Args:
			metric_type (str): 'median' | 'mean' | 'mac_mean'
			id_lists (list): [[idx1, ...], ...]
		Returns:
			(float, float)
		"""
		def cal_mac_mean(a, id_lists):
			return np.mean([np.mean(a[id_list]) for id_list in id_lists])
		if metric_type == 'median':
			return cal_boot_conf_int(x, lambda a: np.median(a), conf_level=conf_level)
			# return cal_hodges_lehmann_median_conf_int(x, conf_level=conf_level)
		elif metric_type == 'mean':
			return cal_boot_conf_int(x, lambda a: np.mean(a), conf_level=conf_level)
		elif metric_type == 'mac_mean':
			assert id_lists is not None
			return cal_boot_conf_int(x, cal_mac_mean, conf_level=conf_level, stat_kwargs={'id_lists': id_lists})
		else:
			raise RuntimeError('Unknown metric type: {}'.format(metric_type))


	def cal_multi_x_mean_stat_conf_interval(self, x_list, metric_type, conf_level=0.95, multi_id_lists=None):
		def cal_mac_mean(a, id_lists):
			return np.mean([np.mean(a[id_list]) for id_list in id_lists])
		if metric_type == 'median':
			return cal_boot_conf_int_for_multi_x(x_list, lambda a_list: np.mean([np.median(a) for a in a_list]), conf_level=conf_level)
		elif metric_type == 'mean':
			return cal_boot_conf_int_for_multi_x(x_list, lambda a_list:np.mean([np.mean(a) for a in a_list]), conf_level=conf_level)
		elif metric_type == 'mac_mean':
			assert multi_id_lists is not None
			return cal_boot_conf_int_for_multi_x(
				x_list,
				lambda a_list:np.mean([cal_mac_mean(a, id_lists) for a, id_lists in zip(a_list, multi_id_lists)]),
				conf_level=conf_level)
		else:
			raise RuntimeError('Unknown metric type: {}'.format(metric_type))


	def get_phenomizer_raw_results_and_patients(self, data_name):

		from core.utils.utils import get_file_list
		def read_tsv(tsv_path):
			df = pd.read_csv(tsv_path, sep='\t', skiprows=3, header=None,
				names=['p-Value', 'Score', 'Disease-Id', 'Disease-Name', 'Gene-Symbols', 'Entrez-IDs', 'Empty-Column'])
			assert len(df) == 8012
			omim_num, orpha_num, decipher_num = 0, 0, 0
			raw_result = []
			for idx, row_se in df.iterrows():
				dis_code, score = row_se['Disease-Id'], row_se['Score']
				if dis_code.startswith('OMIM:'):
					omim_num += 1
				elif dis_code.startswith('ORPHANET:'):
					orpha_num += 1; dis_code = 'ORPHA:'+dis_code.split(':').pop()
				elif dis_code.startswith('DECIPHER'):
					decipher_num += 1; continue
				else:
					assert False
				raw_result.append((dis_code, score))

			print('{}: \n Len(raw_result) = {}; OMIM = {}; ORPHANET = {}; DECIPHER = {}'.format(
				tsv_path, len(raw_result), omim_num, orpha_num, decipher_num))
			return raw_result

		def get_raw_results(data_folder):
			phe_txts = sorted(get_file_list(data_folder, lambda p:p.endswith('phe.txt')))
			all_ranks = sorted([int(os.path.split(phe_txt)[1].split('-')[0]) for phe_txt in phe_txts])
			raw_results, patients = [], []
			for r in all_ranks:
				print('reading {}'.format(os.path.join(data_folder, f'{r}-phe.txt')))
				hpo_list = open(os.path.join(data_folder, f'{r}-phe.txt')).read().strip().splitlines()
				dis_list = open(os.path.join(data_folder, f'{r}-dis.txt')).read().strip().splitlines()
				patients.append([hpo_list, dis_list])
				raw_result = read_tsv(os.path.join(data_folder, f'{r}.tsv'))
				raw_results.append(raw_result)
			return raw_results, patients
		data_folder = os.path.join(DATA_PATH, 'raw', 'phenomizer_sample_100', data_name)
		raw_results, patients = get_raw_results(data_folder)
		return raw_results, patients


	def process_phenomizer_results(self, metric_set=None):
		def check(patients1, patients2):
			assert len(patients1) == len(patients2)
			pa_num = len(patients1)
			for i in range(pa_num):
				p1, p2 = patients1[i], patients2[i]
				assert set(p1[0]) == set(p2[0])
				assert set(p1[1]) == set(self.remove_dis_map_general(p2[1], use_rd_code=False))
		data_names = ['RAMEDIS', 'CJFH', 'PUMC', 'MME']
		save_model_name = 'Phenomizer'
		for data_name in data_names:
			save_data_name = f'{data_name}_SAMPLE_100'
			patients = self.get_phenomizer_test_data(save_data_name)
			print('handling {}'.format(save_data_name))

			raw_results, patients_tmp = self.get_phenomizer_raw_results_and_patients(save_data_name)
			check(patients, patients_tmp)
			metric_dict = self.get_performance(raw_results, patients, metric_set)
			for k, v in metric_dict.items():
				print(k, v)
			self.save_metric_dict(save_model_name, save_data_name, metric_dict)


	def remove_dis_map_general(self, dis_codes, use_rd_code=None):
		keep_to_remove = DataHelper(self.hpo_reader).keep_dis_to_remove
		use_rd_code = use_rd_code if use_rd_code is not None else self.use_rd_code
		if use_rd_code:
			source_to_rd = self.get_rd_reader().get_source_to_rd()
			keep_to_remove = {source_to_rd[d1]: [source_to_rd[d2] for d2 in d2_list] for d1, d2_list in keep_to_remove.items()}
		rm_dis_codes = set()
		for dis_code in dis_codes:
			if dis_code in keep_to_remove:
				rm_dis_codes.update(keep_to_remove[dis_code])
		return [dis_code for dis_code in dis_codes if dis_code not in rm_dis_codes]


if __name__ == '__main__':
	from core.reader import HPOIntegratedDatasetReader
	keep_dnames = ['OMIM', 'ORPHA', 'CCRD']
	mt = ModelTestor(hpo_reader=HPOIntegratedDatasetReader(keep_dnames=keep_dnames), keep_general_dis_map=False)



