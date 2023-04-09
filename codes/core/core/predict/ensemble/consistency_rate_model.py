import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from collections import Counter
from core.reader.hpo_reader import HPOReader

class ConsistencyRateModel(object):
	def __init__(self, hpo_reader=None):
		super(ConsistencyRateModel, self).__init__()
		hpo_reader = hpo_reader or HPOReader()
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.dis_map_rank = hpo_reader.get_dis_map_rank()
		self.dis_list = hpo_reader.get_dis_list()


	def cal_multi_model_consistency(self, models_raw_result, topk):
		"""
		Args:
			models_raw_result (list): [raw_result1, raw_result2, ...]; raw_result=[(dis_code, score), ...]
			k: int
		Returns:
			dict: {dis_code: consistencyRate if consistencyRate > 0}
		"""
		counter = Counter()    # {dis_code: count}
		for r in models_raw_result:
			counter.update([r[i][0] for i in range(topk)])
		model_num = len(models_raw_result)
		return {dis_code:recallTimes/model_num for dis_code, recallTimes in counter.items()}


	def rerank_raw_result(self, raw_result, models_raw_result, topk=20, threshold=1.0):
		"""
		Args:
			raw_result (list):  [(dis_code, score), ...]
			models_raw_result (list): [raw_result1, raw_result2, ...]; raw_result=[(dis_code, score), ...]
			topk (int)
		Returns:
			list: [(dis_code, score), ...]
		"""
		dis_code_to_cons = self.cal_multi_model_consistency(models_raw_result, topk)
		if threshold <= 0.0:
			accept_dis_codes = set(self.dis_list)
		else:
			accept_dis_codes = set([dis_code for dis_code, consRate in dis_code_to_cons.items() if consRate >= threshold])

		accept, reject = [], []
		for dis_code, score in raw_result:
			if dis_code in accept_dis_codes:
				accept.append((dis_code, score))
			else:
				reject.append((dis_code, score))
		return accept + reject


	def rerank_raw_results_wrapper(self, args):
		raw_result, models_raw_result, topk, threshold = args
		return self.rerank_raw_result(raw_result, models_raw_result, topk, threshold)


	def rerank_raw_results(self, raw_results, models_raw_results, topk=20, threshold=1.0, cpu_use=8):
		"""
		Args:
			raw_results (list): [[(dis_code, score), ...], ...]
			models_raw_results (list): [raw_results1, raw_results2, ...]; raw_results=[[(dis_code, score), ...], ...], len(raw_results) = patient_num
			weight (np.ndarray): len = model_num
		Returns:
			list: [[(dis_code, score), ...], ...]
		"""
		model_num = len(models_raw_results)
		pa_num = len(models_raw_results[0])
		chunk_size = max(min(int(pa_num / cpu_use), 200), 10)
		para_list = [(raw_results[pa_idx], [models_raw_results[modelIdx][pa_idx] for modelIdx in range(model_num)], topk, threshold) for pa_idx in range(pa_num)]
		if cpu_use > 1:
			with Pool(cpu_use) as pool:
				return [raw_result for raw_result in tqdm(pool.imap(self.rerank_raw_results_wrapper, para_list, chunksize=chunk_size), total=len(para_list), leave=False)]
		else:
			return [self.rerank_raw_results_wrapper(para) for para in para_list]

