import os
import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

from core.predict.sim_model.mica_model import MICAModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import DATA_PATH, PHELIST_REDUCE, SET_SIM_SYMMAX, NPY_FILE_FORMAT
from core.utils.utils import binary_search, ret_same, check_load_save, delete_redundacy


class MICAPValueModel(MICAModel):
	def __init__(self, hpo_reader, phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
			noise_reductor=None, keep_k_func=ret_same, model_name=None):
		super(MICAPValueModel, self).__init__(hpo_reader, phe_list_mode, set_sim_method, noise_reductor, keep_k_func)
		self.name = 'MICAPValueModel' if model_name is None else model_name

		self.MIN_QUERY_TERM = 3
		self.MAX_QUERY_TERM = 30
		self.MC_NUM = 10000
		self.MC_SCORE_PATH = DATA_PATH + '/preprocess/model/MICAPValueModel/MC-{}-{}.npy'.format(phe_list_mode, set_sim_method)
		self.mc_score_mat = None
		self.p_value_array = None


	def gen_random_patients(self, phe_len, hpo_list, hpo_dict):
		while True:
			p = random.sample(hpo_list, phe_len)
			p = delete_redundacy(p, hpo_dict)
			if len(p) == phe_len:
				return p


	def train(self):
		super(MICAPValueModel, self).train()
		self.get_mc_score_mat()
		self.p_value_array = np.array([(self.MC_NUM-i)/self.MC_NUM for i in range(self.MC_NUM)]+[0])


	def run_mc_multi_func(self, paras):
		q_len, mc_rank = paras
		return q_len, mc_rank, self.cal_score(self.gen_random_patients(q_len, self.hpo_list, self.hpo_dict))


	@check_load_save('mc_score_mat', 'MC_SCORE_PATH', NPY_FILE_FORMAT)
	def get_mc_score_mat(self, cpu_use=12):
		dis_num = self.hpo_reader.get_dis_num()
		mc_score_mat = np.zeros(shape=(self.MAX_QUERY_TERM, dis_num, self.MC_NUM), dtype=np.float32)
		for q_len in range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM+1):
			print('q_len =', q_len)
			para_list = [(q_len, mc_rank) for mc_rank in range(self.MC_NUM)]
			with Pool(cpu_use) as pool:
				for q_num, mc_rank, scoreArray in tqdm(pool.imap_unordered(self.run_mc_multi_func, para_list, chunksize=100), total=len(para_list), leave=False):
					mc_score_mat[q_num-self.MIN_QUERY_TERM, :, mc_rank] = scoreArray
		mc_score_mat = np.sort(mc_score_mat, axis=-1)
		return mc_score_mat


	def pvalue_correct(self, pvals, method='fdr_bh'):
		"""
		Args:
			pvals (list): p-value list
			method (str): 'bonferroni' | 'fdr_bh' | 'fdr_by' | 'holm' | 'hommel' | ...
		Returns:
			list: p-value list
		"""
		reject, pvals_correct, _, _ =  multipletests(pvals, alpha=0.05, method=method)
		return pvals_correct


	def query(self, phe_list, topk=10):
		"""
		Args:
			phe_list (list): list of phenotype
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		result = super(MICAPValueModel, self).query(phe_list, topk)
		q_num = self.MIN_QUERY_TERM if len(phe_list) < self.MIN_QUERY_TERM else (self.MAX_QUERY_TERM if len(phe_list) > self.MAX_QUERY_TERM else len(phe_list))
		for i in range(len(result)):
			dis, score = result[i]
			p_rank = binary_search(self.mc_score_mat[q_num-self.MIN_QUERY_TERM, self.dis_map_rank[dis], :], score, 0, self.MC_NUM)
			result[i] = (-self.p_value_array[p_rank], score, dis)
		return [(dis, -negPValue) for negPValue, score, dis in sorted(result, reverse=True)]


def generate_model(hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX,
		noise_reductor=None, keep_k_func=ret_same, model_name=None):
	"""
	Returns:
		MICAPValueModel
	"""
	model = MICAPValueModel(hpo_reader, phe_list_mode)
	model.train()
	return model


if __name__ == '__main__':
	model = generate_model()
