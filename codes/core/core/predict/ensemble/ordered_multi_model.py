import heapq
import numpy as np
from copy import deepcopy

from core.predict.model import Model
from core.reader.hpo_reader import HPOReader

from core.utils.cycommon import to_rank_score

class OrderedMultiModel(Model):
	def __init__(self, model_inits=None, hpo_reader=HPOReader(), model_name=None, model_list=None, keep_raw_score=True):
		"""
		Args:
			model_inits (list or None): [(modelInitializer, args, kwargs), ...]
			model_list (list or None)
			hpo_reader:
			model_name:
		"""
		super(OrderedMultiModel, self).__init__()
		self.name = model_name or 'OrderedMultiModel'
		self.model_list = model_list or [init_func(*init_args, **init_kwargs) for init_func, init_args, init_kwargs in model_inits]
		self.MODEL_NUM = len(self.model_list)
		self.keep_raw_score = keep_raw_score

		self.hpo_reader = hpo_reader
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.dis_list = hpo_reader.get_dis_list()


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		raw_score_vecs = [model.query_score_vec(phe_list) for model in self.model_list]
		self.raw_score_mats = [np.reshape(score_vec, (1, -1)) for score_vec in raw_score_vecs]
		return self.combine_score_vecs(raw_score_vecs)


	def combine_score_vecs(self, score_vecs):
		"""
		Args:
			score_vecs (list): [score_vec, ...],
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		m = np.vstack(score_vecs)
		if m.dtype != np.float64:
			m = m.astype(np.float64)
		arg_mat = np.argsort(m).astype(np.int32)
		to_rank_score(m, arg_mat)
		return m.sum(axis=0)


	def query_score_mat(self, phe_lists, chunk_size=200, cpu_use=12):
		raw_score_mats = [model.query_score_mat(phe_lists, chunk_size=chunk_size, cpu_use=cpu_use) for model in self.model_list]
		if self.keep_raw_score:
			self.raw_score_mats = deepcopy(raw_score_mats)
		print('OrderedMultiModel: combining scores...')
		ret_mat = []
		for i in range(len(phe_lists)):
			ret_mat.append(self.combine_score_vecs([score_mat[i] for score_mat in raw_score_mats]))
		return np.vstack(ret_mat)


	def score_vec_to_result(self, score_vec, topk, pa_idx=0):
		"""
		Args:
			score_vec (np.ndarray): (dis_num,)
			topk (int or None):
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		if self.keep_raw_score:
			if topk == None:
				dis_int_scores = sorted([(i, score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
			else:
				dis_int_scores = heapq.nlargest(topk, [(i, score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])
			return [(self.dis_list[i], self.raw_score_mats[0][pa_idx][i]) for i, _ in dis_int_scores]
		else:
			if topk == None:
				return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
			return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])


	def score_mat_to_results(self, phe_lists, score_mat, topk):
		return [
			self.score_vec_to_result(score_mat[i], topk, pa_idx=i) if len(phe_lists[i]) != 0 else self.query_empty(topk)
			for i in range(score_mat.shape[0])
		]



if __name__ == '__main__':

	from core.predict.sim_model.sim_term_overlap_model import generate_sim_TO_q_reduce_model
	from core.predict.ensemble.random_model import RandomModel
	model = OrderedMultiModel([generate_sim_TO_q_reduce_model, RandomModel])

