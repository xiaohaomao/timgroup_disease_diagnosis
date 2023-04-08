import os
import numpy as np
import scipy.sparse as sp
import heapq
import random

from core.reader.hpo_reader import HPOReader
from core.predict.model import SparseVecModel
from core.utils.constant import PHELIST_ANCESTOR, MODEL_PATH, TRAIN_MODE, PREDICT_MODE
from core.helper.data.data_helper import DataHelper
from core.utils.utils import item_list_to_rank_list, delete_redundacy

class RBPModel(SparseVecModel):
	def __init__(self, hpo_reader=HPOReader(), alpha=0.01, mode=TRAIN_MODE, model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(RBPModel, self).__init__(hpo_reader)
		self.name = model_name or 'RBPModel'
		self.init_save_path()
		self.alpha = alpha
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		dis_vec_mat = DataHelper(self.hpo_reader).get_train_X(PHELIST_ANCESTOR, dtype=np.float32)
		dis_vec_mat = dis_vec_mat.multiply(1 / dis_vec_mat.sum(axis=0)).tocsr()
		dis_vec_mat[dis_vec_mat > self.alpha] = self.alpha
		self.dis_vec_mat = dis_vec_mat


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return self.cal_score_for_phe_matrix(phe_matrix)


	def cal_score_for_phe_matrix(self, phe_matrix):
		return self.cal_dot_product(phe_matrix)


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			scipy.sparse.csr.csr_matrix: shape=[1, hpo_num]
		"""
		rank_list = item_list_to_rank_list(phe_list, self.hpo_map_rank)
		m = sp.csr_matrix(([1]*len(rank_list), ([0]*len(rank_list), rank_list)), shape=(1, self.HPO_CODE_NUMBER))
		return m


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


	def query(self, phe_list, topk=10):
		return super(RBPModel, self).query(phe_list, topk)


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'RBPModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_VEC_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_vec_mat.npz')


	def save(self):
		sp.save_npz(self.DIS_VEC_MAT_NPZ, self.dis_vec_mat)


	def load(self):
		self.dis_vec_mat = sp.load_npz(self.DIS_VEC_MAT_NPZ)


class RBPDominantRandomModel(RBPModel):
	def __init__(self, hpo_reader=HPOReader(), alpha=0.01, model_name=None, init_para=True):
		super(RBPDominantRandomModel, self).__init__(hpo_reader, alpha, model_name, init_para=False)
		self.name = model_name or 'RBPDominantRandomModel'
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]


	def cal_score(self, phe_list):
		score_vec = super(RBPDominantRandomModel, self).cal_score(phe_list)
		return list(zip(score_vec, [random.random() for _ in range(len(score_vec))]))


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			ret = sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]), reverse=True)
		else:
			ret = heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]))
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in ret]


if __name__ == '__main__':
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	model = RBPModel(hpo_reader)
