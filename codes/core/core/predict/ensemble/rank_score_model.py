import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from core.reader.hpo_reader import HPOReader
from core.utils.cycommon import to_rank_score
from core.predict.ensemble.random_model import RandomModel

class RankScoreModel(object):
	def __init__(self, hpo_reader=None, seed=777):
		hpo_reader = hpo_reader or HPOReader()
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.dis_map_rank = hpo_reader.get_dis_map_rank()
		self.dis_list = hpo_reader.get_dis_list()
		self.seed = seed




	def raw_result_to_rank_score_vec(self, raw_result):
		rank_score_vec = np.zeros(self.DIS_NUM, dtype=np.float64)
		for i, (dis_code, score) in enumerate(raw_result):
			rank_score_vec[ self.dis_map_rank[dis_code] ] = (self.DIS_NUM - i)/self.DIS_NUM
		return rank_score_vec


	def combine_raw_results(self, models_raw_result, weight=None, random_=True, combine_method='ave', topk=None):
		"""
		Args:
			models_raw_result (list): [raw_result1, raw_result2, ...]; raw_result=[(dis_code, score), ...]
			weight (np.ndarray): len = model_num
			combine_method (str): 'ave' | 'median'
		Returns:
			list: [(dis_code, score), ...]
		"""
		m = np.vstack([self.raw_result_to_rank_score_vec(raw_result) for raw_result in models_raw_result])

		if combine_method == 'ave':
			model_num = len(models_raw_result)
			weight = weight or np.ones((model_num, 1), dtype=np.float64)
			weight /= weight.sum()
			score_vec = (m * weight).sum(axis=0)
		else:
			assert combine_method == 'median'
			score_vec = np.median(m, axis=0)
		if random_:
			score_vec_to_sort = self.combine_with_random(score_vec)
			tmp = sorted([(i, score) for i, score in enumerate(score_vec_to_sort)], key=lambda item: item[1], reverse=True)
			r = [(self.dis_list[i], score_vec[i]) for i, _ in tmp]
		else:
			r = sorted([(self.dis_list[i], score) for i, score in enumerate(score_vec)], key=lambda item: item[1], reverse=True)
		return r if topk is None else r[:topk]

	def combine_raw_results_wrapper(self, args):
		models_raw_result, weight, random_, combine_method = args
		return self.combine_raw_results(models_raw_result, weight, random_, combine_method)


	def combine_many_raw_results(self, models_raw_results, weight=None, random_=True, cpu_use=8, combine_method='ave'):
		"""
		Args:
			models_raw_results (list): [raw_results1, raw_results2, ...]; raw_results=[[(dis_code, score), ...], ...], len(raw_results) = patient_num
			weight (np.ndarray): len = model_num
		Returns:
			list: [[(dis_code, score), ...], ...]
		"""
		print('Combine method: {}; Cpu = {}'.format(combine_method, cpu_use))
		model_num = len(models_raw_results)
		pa_num = len(models_raw_results[0])
		chunk_size = max(min(int(pa_num/cpu_use), 200), 10)

		para_list = [([models_raw_results[modelIdx][pa_idx] for modelIdx in range(model_num)], weight, random_, combine_method) for pa_idx in range(pa_num)]
		if cpu_use > 1:
			with Pool(cpu_use) as pool:
				return [raw_result for raw_result in tqdm(pool.imap(self.combine_raw_results_wrapper, para_list, chunksize=chunk_size), total=len(para_list), leave=False)]
		else:
			return [self.combine_raw_results_wrapper(para) for para in para_list]


	def combine_with_random(self, score_vec):
		"""
		Args:
			score_vecs (list): [score_vec, ...],
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		score_vecs = [score_vec, np.random.random(score_vec.shape[0])]
		m = np.vstack(score_vecs)
		if m.dtype != np.float64:
			m = m.astype(np.float64)
		arg_mat = np.argsort(m).astype(np.int32)
		to_rank_score(m, arg_mat)
		return m.sum(axis=0)
