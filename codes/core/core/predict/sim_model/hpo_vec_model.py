import numpy as np
from core.utils.constant import VEC_COMBINE_MEAN, SET_SIM_SYMMAX, PHELIST_ANCESTOR
from core.utils.utils import vec_combine, item_list_to_rank_list, mat_l2_norm

from core.predict.model import DenseVecModel, ScoreMatModel
from core.reader.hpo_reader import HPOReader

# =========================================================================================================
class DisVecModel(DenseVecModel):
	def __init__(self, hpo_vec_mat, hpo_reader, phe_list_mode, vec_combine_mode):
		super(DisVecModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'DisVecModel'
		self.hpo_vec_mat = hpo_vec_mat  # np.ndarray; shape=(hpo_num, vec_size)
		self.vec_size = self.hpo_vec_mat.shape[1]
		self.vec_combine_mode = vec_combine_mode


	def cal_dis_vec_mat(self):
		self.dis_vec_mat = np.zeros(shape=(self.DIS_CODE_NUMBER, self.vec_size))
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int(self.phe_list_mode)
		for i in range(self.DIS_CODE_NUMBER):
			self.dis_vec_mat[i] = vec_combine(self.hpo_vec_mat, dis_int_to_hpo_int[i], self.vec_combine_mode)


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.ndarray: shape=[1, embed_size]
		"""
		hpo_ranks = item_list_to_rank_list(phe_list, self.hpo_map_rank)
		phe_matrix = vec_combine(self.hpo_vec_mat, hpo_ranks, self.vec_combine_mode)
		phe_matrix = phe_matrix.reshape(1, phe_matrix.shape[0])
		return phe_matrix


# =========================================================================================================
class DisVecEuclideanModel(DisVecModel):
	def __init__(self, hpo_vec_mat, hpo_reader, phe_list_mode, model_name=None, vec_combine_mode=VEC_COMBINE_MEAN):
		super(DisVecEuclideanModel, self).__init__(hpo_vec_mat, hpo_reader, phe_list_mode, vec_combine_mode)
		self.name = 'DisVecEuclideanModel' if model_name is None else model_name


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return -self.cal_euclid_dist(phe_matrix)


# =========================================================================================================
class DisVecCosineModel(DisVecModel):
	def __init__(self, hpo_vec_mat, hpo_reader, phe_list_mode, model_name=None, vec_combine_mode=VEC_COMBINE_MEAN):
		super(DisVecCosineModel, self).__init__(hpo_vec_mat, hpo_reader, phe_list_mode, vec_combine_mode)
		self.name = 'DisVecCosineModel' if model_name is None else model_name


	def cal_score(self, phe_list):
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return self.cal_dot_product(phe_matrix)

	def cal_dis_vec_mat(self):
		super(DisVecCosineModel, self).cal_dis_vec_mat()
		self.mat_l2_norm(self.dis_vec_mat)


	def phe_list_to_matrix(self, phe_list):
		phe_matrix = super(DisVecCosineModel, self).phe_list_to_matrix(phe_list)
		return self.mat_l2_norm(phe_matrix)


# =========================================================================================================
class HPOVecCosineScoreMatModel(ScoreMatModel):
	def __init__(self, hpo_vec_mat, hpo_reader, phe_list_mode, model_name=None, set_sim_method=SET_SIM_SYMMAX):
		super(HPOVecCosineScoreMatModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method)
		self.name = 'HPOVecCosineScoreMatModel' if model_name is None else model_name
		self.hpo_vec_mat = hpo_vec_mat
		self.vec_size = self.hpo_vec_mat.shape[1]
		super(HPOVecCosineScoreMatModel, self).train()


	def cal_score_mat(self):
		self.hpo_norm_mat = mat_l2_norm(self.hpo_vec_mat)
		self.score_mat = np.dot(self.hpo_norm_mat, self.hpo_norm_mat.T)




