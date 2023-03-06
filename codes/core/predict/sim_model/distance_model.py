

from core.predict.model import ScoreMatModel
from core.utils.utils import cal_shortest_dist, cal_mean_shortest_turn_dist
from core.utils.constant import DIST_MEAN_TURN, DIST_SHORTEST, DATA_PATH
import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from core.reader.hpo_reader import HPOReader
from core.utils.constant import SET_SIM_SYMMAX, SET_SIM_EMD, PHELIST_REDUCE

class DistanceModel(ScoreMatModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, dist_type=DIST_SHORTEST,
			set_sim_method=SET_SIM_SYMMAX, init_cpu_use=12, model_name=None, init_para=True):
		"""
		Args:
			hpo_dict (dict):
			dis2hpo (dict):
			distName (str): DIST_SHORTEST | DIST_MEAN_TURN
		"""
		super(DistanceModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method)
		self.name = 'DistanceModel' if model_name is None else model_name
		self.init_cpu_use = init_cpu_use
		PREPROCESS_FOLDER = DATA_PATH + '/preprocess/model/DistanceModel'
		self.SCOREMAT_PATH = PREPROCESS_FOLDER+'/{}_SCOREMAT.npy'.format(dist_type)

		if dist_type == DIST_MEAN_TURN:
			self.dist_func = cal_mean_shortest_turn_dist
		else:   # dist_type == DIST_SHORTEST
			self.dist_func = cal_shortest_dist
		if init_para:
			self.train()


	def cal_score_mat_multi_func(self, paras):
		i, j = paras
		return (i, j, -self.dist_func(self.hpo_list[i], self.hpo_list[j], self.hpo_dict))


	def cal_score_mat(self):
		if os.path.exists(self.SCOREMAT_PATH):
			self.score_mat = np.load(self.SCOREMAT_PATH)
			return
		self.score_mat = np.zeros(shape=[self.HPO_CODE_NUMBER, self.HPO_CODE_NUMBER])
		para_list = [(i, j) for i in range(self.HPO_CODE_NUMBER) for j in range(i, self.HPO_CODE_NUMBER)]
		with Pool(self.init_cpu_use) as pool:
			for i, j, sim in tqdm(
				pool.imap_unordered(self.cal_score_mat_multi_func, para_list, chunksize=int(len(para_list)/self.init_cpu_use/5)+1),
				total=len(para_list), leave=False
			):
				self.score_mat[i, j] = sim
				self.score_mat[j, i] = sim
		np.save(self.SCOREMAT_PATH, self.score_mat)


if __name__ == '__main__':
	model = DistanceModel()


