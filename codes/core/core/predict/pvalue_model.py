import heapq
from tqdm import tqdm
import os
from statsmodels.stats.multitest import multipletests
from scipy.sparse import save_npz, load_npz
import random
import numpy as np

from core.utils.constant import DATA_PATH, SORT_S_P, SORT_P_S, SORT_P, MODEL_PATH
from core.utils.utils import data_to_01_matrix, bise_mat, delete_redundacy, item_list_to_rank_list


class PValueModel(object):
	def __init__(self, model, mc=10000, sort_type=SORT_S_P, pcorrect='fdr_bh'):
		super(PValueModel, self).__init__()
		self.model = model
		self.MC_SAMPLE_NUM = mc
		self.sort_type = sort_type
		self.pcorrect = pcorrect

		self.init_save_folder()

		self.hpo_reader = model.hpo_reader
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		self.dis_map_rank = self.hpo_reader.get_dis_map_rank()

		self.MIN_QUERY_TERM = 1
		self.MAX_QUERY_TERM = 30


	def init_save_folder(self):
		mc_folder = MODEL_PATH + '/PValueModel/MC-{}'.format(self.MC_SAMPLE_NUM)
		self.SAVE_FOLDER = mc_folder + '/{}'.format(self.model.name)
		self.RANDOM_PATIENT_FOLDER = mc_folder + '/RandomPatient'
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		os.makedirs(self.RANDOM_PATIENT_FOLDER, exist_ok=True)


	def result_to_presult(self, phe_list, result):
		raise NotImplementedError


	def get_qlen(self, phe_list):
		return len(delete_redundacy(phe_list, self.hpo_dict))


	def gen_random_patients(self, phe_len, hpo_list, hpo_dict):
		while True:
			p = random.sample(hpo_list, phe_len)
			p = delete_redundacy(p, hpo_dict)
			if len(p) == phe_len:
				return p


	def gen_random_pa(self):
		for phe_len in range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM + 1):
			self.get_random_pa_mat(phe_len)


	def get_random_pa_mat(self, phe_len):
		"""
		Returns:
			csr_matrix: shape=(mcSampleNum, hpo_num); dtype=np.bool_
		"""
		mat_Path = self.RANDOM_PATIENT_FOLDER + '/q_len-{}.npz'.format(phe_len)
		if os.path.exists(mat_Path):
			return load_npz(mat_Path)
		hpo_num = self.hpo_reader.get_hpo_num()
		hpo_int_list = list(range(hpo_num))
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		pa_hpo_int_lists = [self.gen_random_patients(phe_len, hpo_int_list, hpo_int_dict) for _ in tqdm(range(self.MC_SAMPLE_NUM))]
		m = data_to_01_matrix(pa_hpo_int_lists, hpo_num, dtype=np.bool_)
		save_npz(mat_Path, m)
		return m


	def pvalue_correct(self, pvals, method='fdr_bh'):
		"""
		Args:
			pvals (list): p-value list
			method (str): 'bonferroni' | 'fdr_bh' | 'fdr_by' | 'holm' | 'hommel' | ...
		Returns:
			list: p-value list
		"""
		if method is None:
			return pvals
		reject, pvals_correct, _, _ =  multipletests(pvals, alpha=0.05, method=method)
		return pvals_correct


	def cal_raw_score_mat(self, pa_mat, cpu_use=20):
		"""
		Returns:
			np.ndarray (int): shape=(DIS_NUM, PATIENT_NUM)
		"""
		sample_num = pa_mat.shape[0]
		hpo_list = self.hpo_reader.get_hpo_list()

		phe_lists = [[hpo_list[dis_int] for dis_int in pa_mat[i].nonzero()[1]] for i in range(sample_num)]
		return self.model.query_score_mat(phe_lists, cpu_use=cpu_use).astype(np.float32).T


	def score_array_to_hist_mat(self, scoreArray, bins=1000):
		"""[left, right); [left, right] for final bin
		Args:
			scoreArray (np.ndarray): shape=(sample_num, )
		Returns:
			np.ndarray: histArray; shape=(bins,)
			np.ndarray: binArray; shape=(bins+1, )
		"""
		return np.histogram(scoreArray, bins=bins)


	def score_mat_to_hist_mat(self, mcScoreMat, bins=1000):
		"""
		Args:
			mcScoreMat (np.ndarray): shape=(dis_num, sample_num)
		Returns:
			np.ndarray: histMat; shape=(dis_num, bins)
			np.ndarray: binMat; shape=(dis_num, bins+1)
		"""
		he = np.apply_along_axis(np.histogram, -1, mcScoreMat, bins=bins)
		return np.array(list(he[:, 0])).astype(np.int32), np.array(list(he[:, 1])).astype(np.float32)


	def get_bins_width(self, bins_mat):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		return bins_mat[:, 1] - bins_mat[:, 0]


	def cal_accu_hist_mat(self, histMat):
		"""
		Args:
			histMat (np.ndarray): shape=(dis_num, bins)
		Returns:
			np.ndarray: shape=shape(dis_num, bins)
		"""
		m = histMat.copy()
		m[:, -1] = 0
		for i in range(m.shape[1]-2, -1, -1):
			m[:, i] = m[:, i] + m[:, i+1]
		return m


	def sort_result(self, dis_list, score_list, p_list, sort_type, topk):
		ret = [(dis, (score, p)) for dis, score, p in zip(dis_list, score_list, p_list)]
		if sort_type == SORT_P:
			key = lambda item: -item[1][1]
		elif sort_type == SORT_P_S:
			key = lambda item: (-item[1][1], item[1][0])
		elif sort_type == SORT_S_P:
			key = lambda item: (item[1][0], -item[1][1])
		else:
			assert False
		if topk is None:
			return sorted(ret, key=key, reverse=True)
		return heapq.nlargest(topk, ret, key=key)


	def q_len_to_rank(self, q_len):
		return max(min(q_len, self.MAX_QUERY_TERM), self.MIN_QUERY_TERM) - self.MIN_QUERY_TERM


	def get_qlen_dim(self):
		return self.MAX_QUERY_TERM - self.MIN_QUERY_TERM + 1


	def result_to_query_score_vec(self, result):
		"""
		Returns:
			shape=(dis_num,)
		"""
		dis_list, score_list = zip(*result)
		dis_int_list = item_list_to_rank_list(dis_list, self.dis_map_rank)
		v = np.zeros(shape=(len(result),), dtype=np.float32)
		v[dis_int_list] = score_list
		return v


	def results_to_query_score_mat(self, results):
		"""
		Returns:
			shape=(sample_num, dis_num)
		"""
		return np.vstack([self.result_to_query_score_vec(result) for result in results])


# ================================================================================================================
class RawPValueModel(PValueModel):
	def __init__(self, model, model_name=None, mc=10000, sort_type=SORT_S_P, pcorrect='fdr_bh'):
		super(RawPValueModel, self).__init__(model, mc=mc, sort_type=sort_type, pcorrect=pcorrect)
		self.name = model_name if model_name is not None else self.get_default_model_name()
		self.MC_SCORE_MAT_FOLDER = self.SAVE_FOLDER + '/mc_score_mat'
		os.makedirs(self.MC_SCORE_MAT_FOLDER, exist_ok=True)
		self.mc_score_mat = None

		self.dis_list = self.hpo_reader.get_dis_list()
		self.dis_num = self.hpo_reader.get_dis_num()


	def set_sort_type(self, sort_type):
		self.sort_type = sort_type
		self.name = self.get_default_model_name()


	def get_default_model_name(self):
		return '{}-{}-{}-{}-{}'.format(self.model.name, 'RAW', self.MC_SAMPLE_NUM, self.sort_type, self.pcorrect)


	def train(self, cpu_use=12):
		self.mc_score_mat = np.zeros(shape=(self.get_qlen_dim(), self.dis_num, self.MC_SAMPLE_NUM), dtype=np.float32)
		for q_len in tqdm(range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM+1)):
			self.mc_score_mat[self.q_len_to_rank(q_len)] = self.get_mc_raw_score_mat(q_len, cpu_use=cpu_use)


	def mc_simulate(self, cpu_use=12):
		for q_len in tqdm(range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM + 1)):
			print('q_len = {}'.format(q_len))
			self.get_mc_raw_score_mat(q_len, cpu_use=cpu_use)


	def get_raw_score_mat_path(self, q_len):
		return self.MC_SCORE_MAT_FOLDER + '/q_len-{}.npy'.format(q_len)


	def get_mc_raw_score_mat(self, q_len, cpu_use=12):
		mat_Path = self.get_raw_score_mat_path(q_len)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		m = self.cal_raw_score_mat(self.get_random_pa_mat(q_len), cpu_use=cpu_use)
		m = np.sort(m, axis=-1)
		np.save(mat_Path, m)
		return m


	def q_score_vec_to_pvalue(self, phe_list, score_vec):
		q_len = self.get_qlen(phe_list)
		m = self.mc_score_mat[self.q_len_to_rank(q_len)]
		p = bise_mat(m, score_vec)
		p = 1 - p / self.MC_SAMPLE_NUM
		p = self.pvalue_correct(p, self.pcorrect)
		return p


	def result_to_presult(self, phe_list, result):
		qscore_vec = self.result_to_query_score_vec(result)
		p = self.q_score_vec_to_pvalue(phe_list, qscore_vec)
		return self.sort_result(self.dis_list, qscore_vec, p, self.sort_type, topk=None)


	def query(self, phe_list, topk=10):
		qscore_vec = self.model.query_score_vec(phe_list)
		p = self.q_score_vec_to_pvalue(phe_list, qscore_vec)
		return self.sort_result(self.dis_list, qscore_vec, p, self.sort_type, topk=topk)


def generate_raw_pvalue_model(model, model_name=None, mc=10000, sort_type=SORT_S_P, pcorrect='fdr_bh'):
	"""
	Returns:
		RawPValueModel
	"""
	pmodel = RawPValueModel(model, model_name=model_name, mc=mc, sort_type=sort_type, pcorrect=pcorrect)
	pmodel.train()
	return pmodel


# ================================================================================================================
class HistPValueModel(RawPValueModel):
	def __init__(self, model, model_name=None, mc=10000, sort_type=SORT_S_P, bins=1000, pcorrect='fdr_bh'):
		super(HistPValueModel, self).__init__(model, model_name=model_name, mc=mc, sort_type=sort_type, pcorrect=pcorrect)
		self.name = model_name if model_name is not None else self.get_default_model_name()

		self.bins = bins
		self.HIST_MAT_FOLDER = self.SAVE_FOLDER + '/HistMat'
		os.makedirs(self.HIST_MAT_FOLDER, exist_ok=True)

		self.bins_width_mat = None
		self.mc_bins_mat = None
		self.mc_hist_mat = None
		self.mc_accu_hist_mat = None


	def get_default_model_name(self):
		return '{}-{}-{}-{}-{}'.format(self.model.name, 'HIST', self.MC_SAMPLE_NUM, self.sort_type, self.pcorrect)


	def train(self, cpu_use=12):
		dis_num = self.hpo_reader.get_dis_num()
		q_len_dim = self.get_qlen_dim()
		self.bins_width_mat = np.zeros(shape=(q_len_dim, dis_num), dtype=np.float32)
		self.mc_bins_mat = np.zeros(shape=(q_len_dim, dis_num, self.bins+1), dtype=np.float32)
		self.mc_hist_mat = np.zeros(shape=(q_len_dim, dis_num, self.bins+2), dtype=np.int32)
		self.mc_accu_hist_mat = np.zeros(shape=(q_len_dim, dis_num, self.bins+3), dtype=np.int32)
		for q_len in tqdm(range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM+1)):
			q_len_rank = self.q_len_to_rank(q_len)
			bins_mat, histMat, accu_hist_mat = self.get_mat(q_len, cpu_use=cpu_use)
			binswidth = self.get_bins_width(bins_mat)
			self.mc_bins_mat[q_len_rank] = bins_mat
			self.mc_hist_mat[q_len_rank, :, 1:self.bins+1] = histMat
			self.mc_accu_hist_mat[q_len_rank, :, 1:self.bins+1] = accu_hist_mat
			self.bins_width_mat[q_len_rank] = binswidth


	def mc_simulate(self, cpu_use=12):
		for q_len in tqdm(range(self.MIN_QUERY_TERM, self.MAX_QUERY_TERM + 1)):
			print('q_len = {}'.format(q_len))
			self.get_mat(q_len, cpu_use=cpu_use)


	def get_HistMatPath(self, q_len):
		return self.HIST_MAT_FOLDER + '/q_len-{}-bins-{}-Hist.npy'.format(q_len, self.bins)

	def get_bins_mat_path(self, q_len):
		return self.HIST_MAT_FOLDER + '/q_len-{}-bins-{}-Bins.npy'.format(q_len, self.bins)

	def get_accu_hist_mat_path(self, q_len):
		return self.HIST_MAT_FOLDER + '/q_len-{}-bins-{}-AccuHist.npy'.format(q_len, self.bins)


	def get_mat(self, q_len, cpu_use=12):
		"""bins, hist, accuHist
		"""
		histPath, binsPath, accu_hist_path = self.get_HistMatPath(q_len), self.get_bins_mat_path(q_len), self.get_accu_hist_mat_path(q_len)
		if os.path.exists(histPath):
			return np.load(binsPath), np.load(histPath), np.load(accu_hist_path)
		raw_score_mat = self.get_mc_raw_score_mat(q_len, cpu_use=cpu_use)
		histMat, bins_mat = self.score_mat_to_hist_mat(raw_score_mat, self.bins)
		accu_hist_mat = self.cal_accu_hist_mat(histMat)
		np.save(histPath, histMat); np.save(binsPath, bins_mat); np.save(accu_hist_path, accu_hist_mat)
		return bins_mat, histMat, accu_hist_mat


	def q_score_vec_to_pvalue(self, phe_list, score_vec):
		q_len = self.get_qlen(phe_list)
		q_len_rank = self.q_len_to_rank(q_len)
		bins_mat, histMat, accu_hist_mat, binwidth = self.mc_bins_mat[q_len_rank], self.mc_hist_mat[q_len_rank], \
			self.mc_accu_hist_mat[q_len_rank], self.bins_width_mat[q_len_rank]
		dis_int_array = np.array(list(range(self.dis_num)))
		r = bise_mat(bins_mat, score_vec) + 1
		bins_mat = np.hstack(
			[(bins_mat[:, 0] - binwidth).reshape((-1, 1)), bins_mat, (bins_mat[:, -1] + binwidth).reshape((-1, 1))])

		p = accu_hist_mat[dis_int_array, r] + histMat[dis_int_array, r - 1] * (score_vec - bins_mat[dis_int_array, r - 1]) / binwidth
		p = p / self.MC_SAMPLE_NUM
		p = self.pvalue_correct(p, self.pcorrect)
		return p


def generate_hist_pvalue_model(model, model_name=None, mc=10000, sort_type=SORT_S_P, bins=1000, pcorrect='fdr_bh'):
	"""
	Returns:
		HistPValueModel
	"""
	pmodel = HistPValueModel(model, model_name=model_name, mc=mc, sort_type=sort_type, bins=bins, pcorrect=pcorrect)
	pmodel.train()
	return pmodel


if __name__ == '__main__':
	rpm = RawPValueModel(None, None)

