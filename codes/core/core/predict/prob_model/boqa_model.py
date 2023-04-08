import scipy.sparse as sp
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from copy import deepcopy
import random

from core.utils.constant import TEMP_PATH, PROJECT_PATH, DATA_PATH, MODEL_PATH
from core.utils.utils import random_string, list_add_tail, item_list_to_rank_list, delete_redundacy
from core.reader.hpo_reader import HPOReader


class BOQAModel(object):
	def __init__(self, hpo_reader=HPOReader(), use_freq=True, dp=1.0, model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterReader)
			dp (float): default prob
		"""
		self.name = 'BOQAModel' if model_name is None else model_name
		self.hpo_reader = hpo_reader
		self.use_freq = 1 if use_freq else 0
		self.dp = dp
		self.TEMP_FOLDER = os.path.join(TEMP_PATH, hpo_reader.name, self.name)
		self.INPUT_FOLDER = os.path.join(self.TEMP_FOLDER, 'input')
		os.makedirs(self.INPUT_FOLDER, exist_ok=True)
		self.OUTPUT_FOLDER = os.path.join(self.TEMP_FOLDER, 'output')
		os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
		self.JAR_PATH = os.path.join(PROJECT_PATH, 'core', 'predict', 'prob_model', 'boqa-master', 'out', 'artifacts', 'boqa_jar', 'boqa.jar')

		PREPROCESS_FOLDER = os.path.join(MODEL_PATH, hpo_reader.name, 'BOQAModel')
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
		self.HPO_OBO_PATH = HPOReader().HPO_OBO_PATH
		self.ANNOTATION_TAB_PATH = os.path.join(PREPROCESS_FOLDER, 'phenotype_annotation_boqa_{}.tab'.format(dp))

		self.HPO_SEP = ','
		self.RESULT_SEP = '===='

		self.DIS_NUM = self.hpo_reader.get_dis_num()
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		if init_para:
			self.train()


	def train(self):
		self.gen_new_anno_tab(self.dp)


	def gen_new_anno_tab(self, default_prob):
		def check():
			for row in rows:
				if len(row) != 15:
					print(len(row), row)
				assert len(row) == 15
			pass
		if os.path.exists(self.ANNOTATION_TAB_PATH):
			return
		rows, col_names = self.hpo_reader.get_boqa_anno_tab_rows(default_prob)
		check()
		with open(self.ANNOTATION_TAB_PATH, 'w') as f:
			f.write('\n'.join(['\t'.join(info_list) for info_list in rows]))


	def gen_input_file(self, phe_lists, filepath=None):
		"""
		Args:
			phe_lists (list): [[hpo_code, ...], ...]
		Returns:
			str: file path
		"""
		if filepath is None:
			filepath = os.path.join(self.INPUT_FOLDER, random_string(32) + '.txt')
		with open(filepath, 'w') as f:
			f.write('\n'.join([self.HPO_SEP.join(hpo_list) for hpo_list in phe_lists]))
		return filepath


	def get_output_path(self, intput_path):
		_, file_name = os.path.split(intput_path)
		return self.OUTPUT_FOLDER + os.sep + file_name


	def get_results(self, output_path, topk):
		"""
		Args:
			output_path (str)
		Returns:
			list: [result1, result2, ...], result=[(dis1, score1), ...], scores decreasing
		"""
		ret = []
		result_str_list = open(output_path).read().split(self.RESULT_SEP)[:-1]
		for resultStr in result_str_list:
			line_info = [line.split('\t') for line in resultStr.strip().split('\n')]
			result = [(dis_code, float(scoreStr)) for scoreStr, dis_code in line_info]
			assert len(result) == (self.DIS_NUM if topk == -1 else topk)
			ret.append(result)
		return ret


	def query_score_vec(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo_code1, ...]
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return np.random.rand(self.hpo_reader.get_dis_num())
		return self.query_score_mat([phe_list], cpu_use=1)[0]


	def query_score_mat(self, phe_lists, chunk_size=None, cpu_use=12):
		"""
		Returns:
			np.ndarray: shape=(sample_num, dis_num)
		"""
		qry_results = self.query_many(phe_lists, topk=None, chunk_size=chunk_size, cpu_use=cpu_use)
		dis_map_rank = self.hpo_reader.get_dis_map_rank()
		score_mat = np.zeros((len(phe_lists), self.DIS_NUM), dtype=np.float64)
		for i, dis_score_list in enumerate(qry_results):
			dis_codes, scores = zip(*dis_score_list)
			cols = item_list_to_rank_list(dis_codes, dis_map_rank)
			score_mat[i][cols] = scores
		return score_mat


	def query(self, phe_list, topk):
		"""
		Args:
			phe_list (list): list of phenotype
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		return self.query_many_multi_wrap(([phe_list], topk))[0]


	def query_many_multi_wrap(self, paras):
		phe_lists, topk = paras
		phe_lists = [delete_redundacy(phe_list, self.hpo_dict) for phe_list in phe_lists]

		fake_pa_ranks = []
		for i, phe_list in enumerate(phe_lists):
			if len(phe_list) == 0:
				phe_list.append('HP:0000118')
				fake_pa_ranks.append(i)

		topk = -1 if topk is None else topk
		input_path = self.gen_input_file(phe_lists)
		output_path = self.get_output_path(input_path)


		print('java -jar {} -o {} -a {} -p {} -d {} -f {} -k {}'.format(
			self.JAR_PATH, self.HPO_OBO_PATH, self.ANNOTATION_TAB_PATH, input_path, output_path, self.use_freq, topk))

		os.system('java -jar {} -o {} -a {} -p {} -d {} -f {} -k {}'.format(
			self.JAR_PATH, self.HPO_OBO_PATH, self.ANNOTATION_TAB_PATH, input_path, output_path, self.use_freq, topk))
		results = self.get_results(output_path, topk)

		if fake_pa_ranks:
			dis_list = deepcopy(self.hpo_reader.get_dis_list())
			for i in fake_pa_ranks:
				random.shuffle(dis_list)
				results[i] = [(dis_code, 0.0) for dis_code in dis_list]

		os.remove(input_path)
		os.remove(output_path)
		return results


	def query_many(self, phe_lists, topk=10, chunk_size=None, cpu_use=12):
		"""
		Args:
			phe_lists (list): [[hpo1, hpo2, ...], ...]
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list: [result1, result2, ...], result=[(dis1, score1), ...], scores decreasing
		"""
		ret = []
		if cpu_use == 1:
			return self.query_many_multi_wrap((phe_lists, topk))
		with Pool(cpu_use) as pool:
			sample_size = len(phe_lists)
			if chunk_size is None:
				chunk_size = max(min(sample_size // cpu_use, 500), 50)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			para_list = [(phe_lists[intervals[i]: intervals[i + 1]], topk) for i in range(len(intervals) - 1)]
			for results in tqdm(pool.imap(self.query_many_multi_wrap, para_list), total=len(para_list), leave=False):
				ret.extend(results)
		assert len(ret) == len(phe_lists)
		return ret


if __name__ == '__main__':
	from core.reader import HPOFilterDatasetReader
	from core.utils.utils import list_find, get_all_ancestors_for_many
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	hpo_dict = hpo_reader.get_slice_hpo_dict()
	model = BOQAModel(hpo_reader, use_freq=True)


