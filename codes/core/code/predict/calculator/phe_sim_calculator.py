

import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_ancestors, check_load_save
from core.utils.constant import DATA_PATH, MODEL_PATH
from core.utils.constant import NPY_FILE_FORMAT
from core.predict.calculator.ic_calculator import get_hpo_IC_vec, get_hpo_IC_dict


class PheSimCalculator(object):
	def __init__(self, hpo_reader):
		self.hpo_reader = hpo_reader
		self.score_mat = None


	def get_phe_sim_mat(self):
		"""
		Returns:
			np.ndarray: shape=(hpo_num, hpo_num)
		"""
		raise NotImplementedError


class PheMICASimCalculator(PheSimCalculator):
	def __init__(self, hpo_reader=HPOReader()):
		super(PheMICASimCalculator, self).__init__(hpo_reader)
		self.SCOREMAT_NPY = os.path.join(MODEL_PATH, hpo_reader.name, 'MICAModel', 'mica_model_score_mat.npy')


	@check_load_save('score_mat', 'SCOREMAT_NPY', NPY_FILE_FORMAT)
	def get_phe_sim_mat(self):
		self.before_cal_score_mat()
		score_mat = np.zeros(shape=[self.HPO_CODE_NUMBER, self.HPO_CODE_NUMBER])
		para_list = [(i, j) for i in range(self.HPO_CODE_NUMBER) for j in range(i, self.HPO_CODE_NUMBER)]
		cpu_use=os.cpu_count()
		with Pool(cpu_use) as pool:
			for i, j, sim in tqdm(
				pool.imap_unordered(self.cal_all_phe_sim_multi_func, para_list, chunksize=int(len(para_list)/cpu_use/5)+1),
				total=len(para_list), leave=False
			):
				score_mat[i, j] = sim
				score_mat[j, i] = sim
		return score_mat


	def before_cal_score_mat(self):
		self.HPO_CODE_NUMBER = self.hpo_reader.get_hpo_num()
		self.hpo_list = self.hpo_reader.get_hpo_list()
		self.hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		self.IC = get_hpo_IC_dict(self.hpo_reader)


	def cal_all_phe_sim_multi_func(self, paras):
		i, j = paras
		return (i, j, self.cal_phe_sim(self.hpo_list[i], self.hpo_list[j]))


	def get_phe_sim(self, hpo_code1, hpo_code2):
		i, j = self.hpo_map_rank[hpo_code1], self.hpo_map_rank[hpo_code2]
		return self.score_mat[i, j]
		# return self.hpoSimList[int(i*self.HPO_CODE_NUMBER+j-i*(i+1)/2)]


	def cal_phe_sim2(self, code1, code2):

		common_ances_set = get_all_ancestors(code1, self.hpo_dict) & get_all_ancestors(code2, self.hpo_dict)
		max_IC = -1
		for code in common_ances_set:
			if self.IC[code] > max_IC and self.IC[code] != np.inf:
				max_IC = self.IC[code]
		return max_IC


	def cal_phe_sim(self, code1, code2):

		code1_parents = get_all_ancestors(code1, self.hpo_dict)
		# print(code1_parents)
		max_IC_code = self.cal_max_IC_code(code2, code1_parents, {})
		return self.IC[max_IC_code]


	def cal_max_IC_code(self, code, codeSets, ICmaxDict):

		if code in ICmaxDict:
			return ICmaxDict[code]
		if code in codeSets and self.IC[code] != np.inf:
			ICmaxDict[code] = code
			return code
		max_IC, max_IC_code = -1, None
		for p_code in self.hpo_dict[code].get('IS_A', []):
			temp_code = self.cal_max_IC_code(p_code, codeSets, ICmaxDict)
			if self.IC[temp_code] > max_IC:
				max_IC, max_IC_code = self.IC[temp_code], temp_code
		ICmaxDict[code] = max_IC_code
		return max_IC_code


class PheMINICSimCalculator(PheSimCalculator):
	def __init__(self, hpo_reader=HPOReader()):
		super(PheMINICSimCalculator, self).__init__(hpo_reader)
		self.SCOREMAT_NPY = os.path.join(MODEL_PATH, hpo_reader.name, 'MinICModel', 'min_ic_model_score_mat.npy'.format(hpo_reader.name))


	@check_load_save('score_mat', 'SCOREMAT_NPY', NPY_FILE_FORMAT)
	def get_phe_sim_mat(self):
		self.before_cal_score_mat()
		score_mat = np.zeros(shape=[self.HPO_CODE_NUMBER, self.HPO_CODE_NUMBER])
		with Pool() as pool:
			for rowIndex, rank_list, value_list in tqdm(
					pool.imap(self.cal_row_phe_sim_multi_func, range(self.HPO_CODE_NUMBER), chunksize=200),
					total=self.HPO_CODE_NUMBER, leave=False):
				score_mat[rowIndex, rank_list] = value_list
				score_mat[rank_list, rowIndex] = value_list
		return score_mat


	def before_cal_score_mat(self):
		hpo_reader = HPOReader()
		self.HPO_CODE_NUMBER = hpo_reader.get_hpo_num()
		self.hpo_int_dict = hpo_reader.get_hpo_int_dict()
		self.hpo_list = hpo_reader.get_hpo_list()
		self.IC_vec = get_hpo_IC_vec(self.hpo_reader, default_IC=0.0) # np.array; shape = [HPONum]


	def cal_row_phe_sim_multi_func(self, hpo_rank):
		"""
		Args:
			hpo_code (str)
		Returns:
			int: rowIndex; hpo_rank
			list: rank_list
			np.array: similarity value list; shape=len(rank_list)
		"""
		ancestor_int_list = list(get_all_ancestors(hpo_rank, self.hpo_int_dict))
		return hpo_rank, ancestor_int_list, self.IC_vec[ancestor_int_list]



if __name__ == '__main__':
	pass

