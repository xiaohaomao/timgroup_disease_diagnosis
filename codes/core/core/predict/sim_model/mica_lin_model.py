import os
import numpy as np
from core.predict.sim_model.mica_model import MICAModel
from core.utils.constant import DATA_PATH, MODEL_PATH, PHELIST_REDUCE, SET_SIM_SYMMAX
from core.reader.hpo_reader import HPOReader
from core.utils.utils import slice_list_with_keep_set
from core.predict.calculator.ic_calculator import get_hpo_IC_dict

class MICALinModel(MICAModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
			slice_no_anno=True, model_name=None, init_para=True):
		super(MICALinModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method, init_para=False)
		self.name = 'MICALinModel' if model_name is None else model_name
		PREPROCESS_FOLDER = os.path.join(MODEL_PATH, hpo_reader.name, 'MICALinModel')
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
		self.LIN_SCOREMAT_PATH = os.path.join(PREPROCESS_FOLDER, 'mica_lin_model_score_mat.npy')
		self.anno_hpo_set = set(self.hpo_reader.get_anno_hpo_list()) if slice_no_anno else None
		if init_para:
			self.train()


	def cal_score_mat(self):
		if os.path.exists(self.LIN_SCOREMAT_PATH):
			self.score_mat = np.load(self.LIN_SCOREMAT_PATH)
			return
		super(MICALinModel, self).cal_score_mat()
		self.IC = get_hpo_IC_dict(self.hpo_reader)
		divide_matrix = np.zeros(shape=[self.HPO_CODE_NUMBER, self.HPO_CODE_NUMBER])
		for i in range(self.HPO_CODE_NUMBER):
			for j in range(i+1, self.HPO_CODE_NUMBER):
				divide_matrix[i, j] = self.IC[self.hpo_list[i]] + self.IC[self.hpo_list[j]]
				divide_matrix[j, i] = divide_matrix[i, j]
		self.score_mat = 2*(self.score_mat/divide_matrix)
		for i in range(self.HPO_CODE_NUMBER):
			self.score_mat[i, i] = 1.0
		np.save(self.LIN_SCOREMAT_PATH, self.score_mat)


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		phe_list = super(MICALinModel, self).process_query_phe_list(phe_list, phe_list_mode, hpo_dict)
		if self.anno_hpo_set is not None:
			phe_list = slice_list_with_keep_set(phe_list, self.anno_hpo_set)
		return phe_list


if __name__ == '__main__':
	from core.utils.utils import list_find, get_all_ancestors_for_many
	hpo_reader = HPOReader()
	model = MICALinModel(hpo_reader, slice_no_anno=True)




