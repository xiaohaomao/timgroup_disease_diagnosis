


import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack

from core.predict.model import SparseVecModel
from core.predict.calculator.phe_sim_calculator import PheMICASimCalculator, PheMINICSimCalculator
from core.utils.constant import DATA_PATH, PHELIST_REDUCE, NPY_FILE_FORMAT, VEC_TYPE_0_1, PHELIST_ANCESTOR
from core.utils.utils import item_list_to_rank_list, load_save, data_to_01_matrix
from core.helper.data.data_helper import DataHelper
from core.reader.hpo_reader import HPOReader


class RDDModel(SparseVecModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(RDDModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'RDDModel' if model_name is None else model_name
		if init_para:
			self.train()


	def train(self):
		self.dis_vec_mat = DataHelper(self.hpo_reader).get_train_X(phe_list_mode=self.phe_list_mode, vec_type=VEC_TYPE_0_1, sparse=True, dtype=np.bool)    # [dis_num, hpo_num]
		self.Si = self.dis_vec_mat.sum(axis=1)    # np.matrix; [dis_num, hpo_num]


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_mat = data_to_01_matrix([item_list_to_rank_list(phe_list, self.hpo_map_rank)], self.HPO_CODE_NUMBER, dtype=np.bool)   # [1, hpo_num]
		intersect_mat = self.dis_vec_mat.multiply(phe_mat)
		n1 = (self.dis_vec_mat - intersect_mat).sum(axis=1) # np.matrix; [dis_num, 1]
		n2 = (vstack([phe_mat] * self.DIS_CODE_NUMBER) - intersect_mat).sum(axis=1)   # np.matrix; [dis_num, 1]
		n = n1 + n2 # np.matrix; [dis_num, 1]
		Su = len(phe_list)
		max_S = np.hstack([self.Si, np.array([[Su]] * self.DIS_CODE_NUMBER)]).max(axis=1)    # np.matrix; [dis_num, 1]
		return (1 - n / max_S).A.flatten()


if __name__ == '__main__':
	from core.utils.utils import list_find
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	model = RDDModel(hpo_reader, phe_list_mode=PHELIST_REDUCE)
	result = model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463'], topk=None)  # OMIM:610253
