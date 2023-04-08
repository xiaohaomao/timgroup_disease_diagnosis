import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from core.predict.model import ScoreMatModel
from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_ancestors, ret_same, item_list_to_rank_list
from core.utils.constant import DATA_PATH, SET_SIM_SYMMAX, PHELIST_REDUCE
from core.predict.calculator.ic_calculator import get_hpo_IC_dict
from core.predict.calculator.phe_sim_calculator import PheMINICSimCalculator

class MinICModel(ScoreMatModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
			model_name=None, init_para=True):
		"""sim(t1, t2) = min(IC(t1), IC(t1)) if hasDirectedPath(t1, t2) else 0
		"""
		super(MinICModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method)
		self.name = 'MinICModel' if model_name is None else model_name
		if init_para:
			self.train()


	def cal_score_mat(self):
		self.score_mat = PheMINICSimCalculator(self.hpo_reader).get_phe_sim_mat()


if __name__ == '__main__':
	from core.utils.utils import list_find
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	model = MinICModel()
