import numpy as np
import os

from core.reader.hpo_reader import HPOReader
from core.predict.sim_model.mica_model import MICAModel
from core.utils.constant import SET_SIM_SYMMAX, PHELIST_REDUCE, DATA_PATH, MODEL_PATH
from core.predict.calculator.phe_sim_calculator import PheMICASimCalculator
from core.utils.utils import slice_list_with_keep_set
from core.predict.calculator.ic_calculator import get_hpo_IC_vec


class MICAJCModel(MICAModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
			slice_no_anno=True, model_name=None, init_para=True):
		super(MICAJCModel, self).__init__(hpo_reader, phe_list_mode=phe_list_mode, set_sim_method=set_sim_method, init_para=False)
		self.name = 'MICAJCModel' if model_name is None else model_name
		PREPROCESS_FOLDER = os.path.join(MODEL_PATH, hpo_reader.name, 'MICAJCModel')
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
		self.JC_SCOREMAT_PATH = os.path.join(PREPROCESS_FOLDER,  'mica_jc_model_score_mat.npy')
		self.anno_hpo_set = set(self.hpo_reader.get_anno_hpo_list()) if slice_no_anno else None
		if init_para:
			self.train()


	def cal_score_mat(self):
		if os.path.exists(self.JC_SCOREMAT_PATH):
			self.score_mat = np.load(self.JC_SCOREMAT_PATH)
			return
		super(MICAJCModel, self).cal_score_mat()
		hpo_IC_vec = get_hpo_IC_vec(self.hpo_reader)
		new_score_mat = np.vstack([hpo_IC_vec] * self.HPO_CODE_NUMBER)
		new_score_mat = new_score_mat + new_score_mat.T
		self.score_mat = 1.0 / (1.0 + new_score_mat - 2 * self.score_mat)
		for i in range(self.HPO_CODE_NUMBER):
			self.score_mat[i, i] = 1.0
		assert np.sum(np.isnan(self.score_mat)) == 0
		np.save(self.JC_SCOREMAT_PATH, self.score_mat)


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		phe_list = super(MICAJCModel, self).process_query_phe_list(phe_list, phe_list_mode, hpo_dict)
		if self.anno_hpo_set is not None:
			phe_list = slice_list_with_keep_set(phe_list, self.anno_hpo_set)
		return phe_list


if __name__ == '__main__':
	from core.utils.utils import list_find, get_all_ancestors_for_many
	hpo_reader = HPOReader()
	model = MICAJCModel(hpo_reader, slice_no_anno=True)





