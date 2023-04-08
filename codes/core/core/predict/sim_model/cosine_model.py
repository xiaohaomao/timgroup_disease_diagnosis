import os
import numpy as np
import scipy.sparse as sp

from core.predict.sim_model.sim_term_overlap_model import SimTOModel
from core.utils.utils import get_all_ancestors_for_many
from core.utils.constant import PHELIST_ANCESTOR, MODEL_PATH, TRAIN_MODE, PREDICT_MODE
from core.reader.hpo_reader import HPOReader


class CosineModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=None, init_para=True):
		"""0-1 vector cosine similarity; ancestor extend
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(CosineModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = model_name or 'CosineModel'
		self.init_save_path()
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def cal_dis_vec_mat(self):
		super(CosineModel, self).cal_dis_vec_mat()
		self.dis_vec_mat = self.mat_l2_norm(self.dis_vec_mat)


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			scipy.sparse.csr.csr_matrix: shape=[1, hpo_num]
		"""
		m = super(CosineModel, self).phe_list_to_matrix(phe_list)
		return self.mat_l2_norm(m)


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'CosineModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_VEC_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'DisVecMat.npz')


	def save(self):
		sp.save_npz(self.DIS_VEC_MAT_NPZ, self.dis_vec_mat)


	def load(self):
		self.dis_vec_mat = sp.load_npz(self.DIS_VEC_MAT_NPZ)


class TestCosineModel(object):
	def __init__(self):
		self.model = CosineModel(HPOReader())
		self.model.train()


	def test_matrix_all_one(self):
		row_col_list = zip(*self.model.dis_vec_mat.nonzero())
		for row, col in row_col_list:
			assert self.model.dis_vec_mat[row, col] == 1


	def test_query(self):
		query_input = ['HP:0001519', 'HP:0008909', 'HP:0100554']
		query_extend = get_all_ancestors_for_many(query_input, self.model.hpo_dict)
		result1_extend = get_all_ancestors_for_many(["HP:0001939", "HP:0004322"], self.model.hpo_dict)
		intersection = query_extend.intersection(result1_extend)
		print('query_extend =', query_extend)
		print('result1_extend =', result1_extend)
		print('intersection =', intersection)
		print('len(query_extend)=%d, len(result0Extend)=%d, len(intersection)=%d, sim=%f' %
			(len(query_extend), len(result1_extend), len(intersection), len(intersection)/(np.sqrt(len(result1_extend)) * np.sqrt(len(query_extend)))))
		print(self.model.query(query_input))


if __name__ == '__main__':
	from core.predict.pvalue_model import generate_raw_pvalue_model, generate_hist_pvalue_model
	from core.reader import HPOFilterDatasetReader


