import numpy as np
from scipy import stats
import heapq
from core.predict.model import Model
from core.utils.constant import PHELIST_ANCESTOR
from core.utils.utils import item_list_to_rank_list, get_all_ancestors_for_many, timer
from core.helper.data.data_helper import data_to_01_dense_matrix
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.ic_calculator import get_hpo_IC_dict


class GDDPWeightTOModel(Model):
	def __init__(self, hpo_reader=HPOReader(), fisher='two-sided', model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
			fisher (str): 'two-sided', 'less', 'greater'
		"""
		super(GDDPWeightTOModel, self).__init__()
		self.name = model_name or 'GDDPWeightTOModel'
		self.hpo_reader = hpo_reader
		self.fisher_method = fisher
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.HPO_NUM = hpo_reader.get_hpo_num()
		self.hpo_int_dict = hpo_reader.get_hpo_int_dict()
		self.hpo_list = hpo_reader.get_hpo_list()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()
		self.dis_list = hpo_reader.get_dis_list()
		if init_para:
			self.train()


	@timer
	def train(self):
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR)
		self.dis_hpo_ances_mat = data_to_01_dense_matrix([dis_int_to_hpo_int[i] for i in range(self.DIS_NUM)], self.HPO_NUM, dtype=np.bool)
		self.not_dis_hpo_ances_mat = ~self.dis_hpo_ances_mat
		IC_dict = get_hpo_IC_dict(self.hpo_reader, default_IC=0)
		self.IC_vec = np.array([IC_dict[hpo_code] for hpo_code in self.hpo_list]).T


	def cal_score(self, phe_int_list):
		phe_ances_int_list = get_all_ancestors_for_many(phe_int_list, self.hpo_int_dict)
		q_hpo_ances_mat = data_to_01_dense_matrix([phe_ances_int_list], self.HPO_NUM, dtype=np.bool)
		a = (self.dis_hpo_ances_mat*q_hpo_ances_mat).dot(self.IC_vec).flatten()
		b = (self.not_dis_hpo_ances_mat*q_hpo_ances_mat).dot(self.IC_vec).flatten()
		c = (self.dis_hpo_ances_mat*(~q_hpo_ances_mat)).dot(self.IC_vec).flatten()
		d = (self.not_dis_hpo_ances_mat*(~q_hpo_ances_mat)).dot(self.IC_vec).flatten()
		score_vec = np.zeros(shape=[self.DIS_NUM], dtype=np.float32)
		for i in range(self.DIS_NUM):
			_, p = stats.fisher_exact([[a[i], b[i]], [c[i], d[i]]], alternative=self.fisher_method)
			score_vec[i] = -p
		return score_vec


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		score_vec = self.cal_score(item_list_to_rank_list(phe_list, self.hpo_map_rank))
		assert np.sum(np.isnan(score_vec)) == 0  #
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )



	def query(self, phe_list, topk=10):
		if len(phe_list) == 0:
			return self.query_empty(topk)
		score_vec = self.query_score_vec(phe_list)
		return self.score_vec_to_result(score_vec, topk)


if __name__ == '__main__':

	pass