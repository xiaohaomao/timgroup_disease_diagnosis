import numpy as np
import heapq

from core.predict.model import SparseVecModel
from core.utils.utils import data_to_01_matrix, item_list_to_rank_list
from core.utils.constant import PHELIST_ANCESTOR, VEC_TYPE_0_1
from core.helper.data.data_helper import DataHelper
from core.reader.hpo_reader import HPOReader


class EuclideanModel(SparseVecModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, model_name=None, init_para=True):
		super(EuclideanModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'EuclideanModel' if model_name is None else model_name
		self.vec_type = vec_type
		if init_para:
			self.train()


	def train(self):
		self.dh = DataHelper(self.hpo_reader)
		self.cal_dis_vec_mat()


	def cal_dis_vec_mat(self):
		self.dis_vec_mat = self.dh.get_train_X(self.phe_list_mode, self.vec_type, sparse=True, dtype=np.int32)


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return -self.cal_euclid_dist2(phe_matrix)


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			scipy.sparse.csr.csr_matrix: shape=[1, hpo_num]
		"""
		return self.dh.col_lists_to_matrix(
			[item_list_to_rank_list(phe_list, self.hpo_map_rank)],
			self.HPO_CODE_NUMBER, dtype=np.int32, vec_type=VEC_TYPE_0_1, sparse=True
		)


if __name__ == '__main__':
	model = EuclideanModel()

