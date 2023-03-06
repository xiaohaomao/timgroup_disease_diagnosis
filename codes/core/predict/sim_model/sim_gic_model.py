

import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack

from core.predict.sim_model.sim_term_overlap_model import SimTOModel
from core.predict.calculator.ic_calculator import get_hpo_IC_dict, get_hpo_IC_vec
from core.reader.hpo_reader import HPOReader
from core.utils.utils import slice_dict_with_keep_set
from core.utils.constant import PHELIST_ANCESTOR, MODEL_PATH, TRAIN_MODE, PREDICT_MODE


class SimGICModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(SimGICModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = model_name or 'SimGICModel'
		self.init_save_path()
		self.IC_vec_T = None  # np.array; shape=[HPONum, 1]
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		super(SimGICModel, self).train()
		self.dis_vec_mat = self.dis_vec_mat.astype(np.bool)
		self.IC_vec_T = get_hpo_IC_vec(self.hpo_reader, 0).T  # np.array; shape = [HPONum, 1]


	def cal_score(self, phe_list):
		"""
		Args:
			phe_matrix (scipy.sparse.csr.csr_matrix): shape=[1, hpo_num]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		duprow_phe_matrix = vstack([phe_matrix]*self.dis_vec_mat.shape[0])
		intersect_sum_IC = self.dis_vec_mat.multiply(duprow_phe_matrix) * self.IC_vec_T # np.array; shape=[dis_num]
		union_sum_IC = (self.dis_vec_mat + duprow_phe_matrix) * self.IC_vec_T   # np.array; shape=[dis_num]
		return (intersect_sum_IC / union_sum_IC).flatten()


	def phe_list_to_matrix(self, phe_list):
		return super(SimGICModel, self).phe_list_to_matrix(phe_list).astype(np.bool)


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'SimGICModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_VEC_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_vec_mat.npz')
		self.IC_VEC_T_NPY = os.path.join(self.SAVE_FOLDER, 'IC_vec_T.npy')

	def save(self):
		sp.save_npz(self.DIS_VEC_MAT_NPZ, self.dis_vec_mat)
		np.save(self.IC_VEC_T_NPY, self.IC_vec_T)


	def load(self):
		self.dis_vec_mat = sp.load_npz(self.DIS_VEC_MAT_NPZ)
		self.IC_vec_T = np.load(self.IC_VEC_T_NPY)


if __name__ == '__main__':
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	model = SimGICModel(hpo_reader)
