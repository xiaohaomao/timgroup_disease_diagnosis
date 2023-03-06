
import os
import numpy as np
import scipy.sparse as sp
from core.predict.sim_model.sim_term_overlap_model import SimTOModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import PHELIST_ANCESTOR, MODEL_PATH, TRAIN_MODE, PREDICT_MODE


class JaccardModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=None, init_para=True):

		super(JaccardModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = model_name or 'JaccardModel'
		self.init_save_path()
		self.phe_size_vec = None # shape=[dis_num]
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		super(JaccardModel, self).train()
		self.phe_size_vec = np.array(self.dis_vec_mat.sum(axis=1)).flatten()    # np.array; shape=[dis_num]


	def cal_score(self, phe_list):
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return self.cal_score_for_phe_matrix(phe_matrix)


	def cal_score_for_phe_matrix(self, phe_matrix):
		score_vec = super(JaccardModel, self).cal_score_for_phe_matrix(phe_matrix)    # np.array; shape=[dis_num]; Overlap Numbers
		return score_vec / (self.phe_size_vec + phe_matrix.count_nonzero() - score_vec)


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'JaccardModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_VEC_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'DisVecMat.npz')
		self.PHE_SIZE_VEC_NPY = os.path.join(self.SAVE_FOLDER, 'PheSizeVec.npy')


	def save(self):
		sp.save_npz(self.DIS_VEC_MAT_NPZ, self.dis_vec_mat)
		np.save(self.PHE_SIZE_VEC_NPY, self.phe_size_vec)


	def load(self):
		self.dis_vec_mat = sp.load_npz(self.DIS_VEC_MAT_NPZ)
		self.phe_size_vec = np.load(self.PHE_SIZE_VEC_NPY)


if __name__ == '__main__':
	from core.utils.utils import list_find
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	model = JaccardModel(hpo_reader)
	result = model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463'], topk=None)  # OMIM:610253
	print(result[:10])
	print(list_find(result, lambda item: item[0] == 'OMIM:610253'))