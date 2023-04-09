from collections import Counter
import numpy as np
import os
from tqdm import tqdm
import json
from multiprocessing import Pool
from core.predict.model import ScoreMatModel
from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_ancestors, delete_redundacy, get_all_ancestors_for_many, ret_same, item_list_to_rank_list
from core.utils.constant import DATA_PATH, SET_SIM_SYMMAX, SET_SIM_EMD, PHELIST_REDUCE, DISEASE_ANNOTATION, GENE_ANNOTATION

from core.predict.calculator.phe_sim_calculator import PheMICASimCalculator


class MICAModel(ScoreMatModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
			model_name=None, init_para=True):

		super(MICAModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method)
		self.name = 'MICAModel' if model_name is None else model_name
		if init_para:
			self.train()


	def cal_distance_mat(self):
		self.cal_score_mat()
		self.distance_mat = np.max(self.score_mat) - self.score_mat
		del self.score_mat


	def cal_score_mat(self):
		self.score_mat = PheMICASimCalculator(self.hpo_reader).get_phe_sim_mat()


if __name__ == '__main__':

	pass