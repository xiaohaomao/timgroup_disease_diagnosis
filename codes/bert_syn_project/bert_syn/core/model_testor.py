

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from bert_syn.utils.utils import get_all_ancestors_for_many, get_all_ancestors, check_return, timer, dict_list_add
from bert_syn.core.data_helper import HPOReader


class ModelTestor(object):
	def __init__(self):
		self.hpo_reader = None


	@check_return('hpo_reader')
	def get_hpo_reader(self):
		return HPOReader()


	def cal_rank(self, raw_term, result, true_text_set, max_rank):
		"""
		Args:
			result (list): [(true_term, score), ...]
		Returns:
		"""
		result = sorted(result, key=lambda item: item[1], reverse=True)
		assert len(result) > 0
		rank = max_rank
		for i, (term, score) in enumerate(result):
			if term in true_text_set:
				rank = i + 1
				break
		return rank, raw_term


	def cal_rank_wrapper(self, paras):
		return self.cal_rank(*paras)


	@timer
	def cal_metrics(self, result, raw_to_true_texts, recall_at_k=None, cpu_use=12, chunksize=100):
		"""
		Args:
			result (str or list):
				str: csv_path, columns=('text_a', 'text_b', 'score')
				list: [(text_a, text_b, score), ...]
			raw_to_true_texts (dict): {raw_term: [true_text1, true_text2, ...]}
			recall_at_k (list): e.g. [1, 10]
		Returns:
			dict: {
				'MEDIAN_RANK': float,
				'RECALL_1': float,
				'RECALL_10': float,
			}
			dict: {
				raw_term: rank
			}
		"""
		def process_result_csv(result_csv):
			df = pd.read_csv(result_csv)
			print('Getting raw_term_to_result...')
			raw_term_to_result = {}  # {raw_term: [(true_term, score), ...]}
			for raw_term, true_term, score in tqdm(df.values):
				dict_list_add(raw_term, (true_term, score), raw_term_to_result)
			return raw_term_to_result

		def process_result_list(samples):
			raw_term_to_result = {}  # {raw_term: [(true_term, score), ...]}
			for raw_term, true_term, score in tqdm(samples):
				dict_list_add(raw_term, (true_term, score), raw_term_to_result)
			return raw_term_to_result

		def get_iterator(raw_term_to_result, raw_to_true_texts, max_rank):
			for raw_term, true_texts in tqdm(raw_to_true_texts.items()):
				true_text_set = set(true_texts)
				yield raw_term, raw_term_to_result[raw_term], true_text_set, max_rank

		recall_at_k = recall_at_k or (1, 10)

		if isinstance(result, str) and result.endswith('.csv'):
			raw_term_to_result = process_result_csv(result)
		elif isinstance(result, list):
			raw_term_to_result = process_result_list(result)
		else:
			raise RuntimeError('Unknown result type: {}'.format(type(result)))

		raw_term_to_rank = {}; rank_list = []
		max_rank = len(self.get_hpo_reader().get_cns_list())
		it = get_iterator(raw_term_to_result, raw_to_true_texts, max_rank)
		with Pool(cpu_use) as pool:
			for rank, raw_term in pool.imap(self.cal_rank_wrapper, it, chunksize=chunksize):
				rank_list.append(rank)
				raw_term_to_rank[raw_term] = rank
		metric_dict = {}
		metric_dict['MEDIAN_RANK'] = np.median(rank_list)
		for k in recall_at_k:
			metric_dict[f'RECALL_{k}'] = sum([r <= k for r in rank_list]) / len(rank_list)
		return metric_dict, raw_term_to_rank



if __name__ == '__main__':
	pass
