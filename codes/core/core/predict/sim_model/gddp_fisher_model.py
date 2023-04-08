import os
import numpy as np
from tqdm import tqdm

from core.predict.model import DenseVecModel
from core.predict.calculator.phe_sim_calculator import PheMICASimCalculator, PheMINICSimCalculator
from core.utils.constant import PHE_SIM_MINIC, PHE_SIM_MICA, PHELIST_REDUCE, NPY_FILE_FORMAT, MODEL_PATH, TRAIN_MODE, PREDICT_MODE
from core.utils.utils import item_list_to_rank_list, load_save
from core.helper.data.data_helper import data_to_01_dense_matrix
from core.reader.hpo_reader import HPOReader


class GDDPFisherModel(DenseVecModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, phe_sim=PHE_SIM_MINIC, gamma=0.5,
			mode=TRAIN_MODE, model_name=None, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(GDDPFisherModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = model_name or 'GDDPFisherModel'
		self.init_save_path()
		self.phe_sim = phe_sim
		self.gamma = gamma

		PREPROCESS_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'GDDPFisherModel')
		self.MICA_SIK_NPY = os.path.join(PREPROCESS_FOLDER, 'gddp_fisher_model_mica_sik.npy')
		self.MICA_PIK_NPY = os.path.join(PREPROCESS_FOLDER, 'gddp_fisher_model_mica_pik.npy')
		self.MINIC_SIK_NPY = os.path.join(PREPROCESS_FOLDER, 'gddp_fisher_model_minic_sik.npy')
		self.MINIC_PIK_NPY = os.path.join(PREPROCESS_FOLDER, 'gddp_fisher_model_minic_pik.npy')

		self.log_pik_mat = None
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		if self.phe_sim == PHE_SIM_MICA:
			sik_mat, log_pik_mat = self.get_MICA_sik(), self.get_MICA_log_pik()
		else:
			sik_mat, log_pik_mat = self.get_min_IC_sik(), self.get_min_IC_log_pik()
		self.log_pik_mat = log_pik_mat * (sik_mat >= self.gamma)


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_mat = data_to_01_dense_matrix([item_list_to_rank_list(phe_list, self.hpo_map_rank)], self.HPO_CODE_NUMBER, dtype=np.float32)
		return -phe_mat.dot(self.log_pik_mat).flatten()


	def get_sik(self, phe_sim):
		sik_mat = np.zeros(shape=(self.HPO_CODE_NUMBER, self.DIS_CODE_NUMBER), dtype=np.float32)
		phe_sim_mat = PheMINICSimCalculator(self.hpo_reader).get_phe_sim_mat() if phe_sim == PHE_SIM_MINIC else PheMICASimCalculator(self.hpo_reader).get_phe_sim_mat()
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int()
		for dis_int, hpo_int_list in tqdm(dis_int_to_hpo_int.items()):
			sik_mat[:, dis_int] = phe_sim_mat[:, hpo_int_list].max(axis=1)
		return sik_mat


	def get_log_pik(self, sik_mat):
		N = self.DIS_CODE_NUMBER
		log_pik_mat = np.log((N - sik_mat.argsort().argsort()) / N)
		return log_pik_mat


	@load_save('MINIC_SIK_NPY', NPY_FILE_FORMAT)
	def get_min_IC_sik(self):
		return self.get_sik(PHE_SIM_MINIC)


	@load_save('MINIC_PIK_NPY', NPY_FILE_FORMAT)
	def get_min_IC_log_pik(self):
		return self.get_log_pik(self.get_min_IC_sik())


	@load_save('MICA_SIK_NPY', NPY_FILE_FORMAT)
	def get_MICA_sik(self):
		return self.get_sik(PHE_SIM_MICA)


	@load_save('MICA_PIK_NPY', NPY_FILE_FORMAT)
	def get_MICA_log_pik(self):
		return self.get_log_pik(self.get_MICA_sik())


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'GDDPFisherModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.PIK_MAT_NPY = os.path.join(self.SAVE_FOLDER, 'PikMat.npy')


	def save(self):
		np.save(self.PIK_MAT_NPY, self.log_pik_mat)


	def load(self):
		self.log_pik_mat = np.load(self.PIK_MAT_NPY)


if __name__ == '__main__':
	from core.utils.utils import list_find
	from core.reader import HPOFilterDatasetReader





