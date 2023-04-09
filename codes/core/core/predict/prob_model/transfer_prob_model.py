from core.predict.model import Model
from core.reader.hpo_reader import HPOReader
from core.utils.utils import list_add_tail, item_list_to_rank_list, get_csr_matrix_from_dict, get_all_ancestors_for_many, delete_redundacy
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE
from scipy.sparse import csr_matrix, vstack
import numpy as np
import heapq


# ======================================================================================================================
class TransferProbModel(Model):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, alpha=0.5, model_name=None, init_para=True):
		super(TransferProbModel, self).__init__()
		self.name = 'TransferProbModel' if model_name is None else model_name
		self.hpo_reader = hpo_reader
		self.dp = default_prob
		self.alpha = alpha
		self.hpo_dict = hpo_reader.get_slice_hpo_dict()
		self.HPO_NUM, self.DIS_NUM = hpo_reader.get_hpo_num(), hpo_reader.get_dis_num()
		self.hpo_map_rank, self.dis_map_rank = hpo_reader.get_hpo_map_rank(), hpo_reader.get_dis_map_rank()
		self.dis_list = hpo_reader.get_dis_list()

		self.missing_rate_mat = None
		self.noise_rate_mat = None
		self.dis_hpo_mat = None
		self.dis_hpo_ances_mat = None
		if init_para:
			self.train()


	def train(self):
		self.cal_missing_rate_mat()
		self.cal_noise_rate_mat()
		self.cal_dis_hpo_mat()
		self.cal_dis_hpo_ances_mat()


	def cal_missing_rate_mat(self):
		dis_to_hpo_prob = self.hpo_reader.get_dis_to_hpo_prob(default_prob=self.dp)
		row, col, data = [], [], []
		for dis_code, hpo_prob_list in dis_to_hpo_prob.items():
			row.extend([self.dis_map_rank[dis_code]] * len(hpo_prob_list))
			hpo_list, prob_list = list(zip(*hpo_prob_list))
			col.extend(item_list_to_rank_list(hpo_list, self.hpo_map_rank))
			data.extend(prob_list)
		data = np.log(1 - np.array(data))
		data[np.isneginf(data)] = np.log(1-0.99)
		self.missing_rate_mat = csr_matrix((data, (row, col)), shape=(self.DIS_NUM, self.HPO_NUM))



	def cal_noise_rate_mat(self):
		hpo2dis = self.hpo_reader.get_hpo_int_to_dis_int(PHELIST_ANCESTOR)
		M = np.zeros(shape=[self.HPO_NUM,], dtype=np.float32)
		for hpo_rank, disRankList in hpo2dis.items():
			M[hpo_rank] = len(disRankList)
		M = np.log(M / self.DIS_NUM)
		M[np.isneginf(M)] = np.log(1/self.DIS_NUM)
		self.noise_rate_mat = M

	def cal_dis_hpo_mat(self):
		self.dis_hpo_mat = get_csr_matrix_from_dict(self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_REDUCE),
											shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True)


	def cal_dis_hpo_ances_mat(self):
		self.dis_hpo_ances_mat = get_csr_matrix_from_dict(self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR),
												shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True)


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		phe_extend_list = get_all_ancestors_for_many(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0:item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM),
			dtype=np.bool, t=True)
		q_hpo_ances_mat = get_csr_matrix_from_dict({0:item_list_to_rank_list(phe_extend_list, self.hpo_map_rank)},
			shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		miss_log_prob = (self.dis_hpo_mat - self.dis_hpo_mat.multiply(q_hpo_ances_mat)).multiply(self.missing_rate_mat).sum(axis=1).getA1()  # np.ndarray; shape=(dis_num, 1)

		noise_log_prob = (vstack([q_hpo_mat] * self.DIS_NUM) - self.dis_hpo_ances_mat.multiply(q_hpo_mat)).multiply(self.noise_rate_mat).sum(axis=1).getA1()  # np.matrix; shape=(dis_num, 1)

		score_vec = self.alpha * miss_log_prob + (1 - self.alpha) * noise_log_prob

		return score_vec


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		score_vec = self.cal_score(phe_list)
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


# ======================================================================================================================
class TransferProbNoisePunishModel(TransferProbModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, model_name=None, init_para=True):
		super(TransferProbNoisePunishModel, self).__init__(hpo_reader, default_prob=default_prob, init_para=False)
		self.name = model_name or 'TransferProbNoisePunishModel'
		if init_para:
			self.train()

	def train(self):
		self.cal_noise_rate_mat()
		self.cal_dis_hpo_ances_mat()


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0:item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		noise_log_prob = (vstack([q_hpo_mat] * self.DIS_NUM) - self.dis_hpo_ances_mat.multiply(q_hpo_mat)).multiply(self.noise_rate_mat).sum(axis=1)  # np.matrix; shape=(dis_num, 1)
		score_vec = noise_log_prob.getA1()
		return score_vec


# ======================================================================================================================
class TransferProbMissPunishModel(TransferProbModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, init_para=True):
		super(TransferProbMissPunishModel, self).__init__(hpo_reader, default_prob, init_para=False)
		self.name = 'TransferProbMissPunishModel'
		if init_para:
			self.train()

	def train(self):
		self.cal_missing_rate_mat()
		self.cal_dis_hpo_mat()


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		phe_extend_list = get_all_ancestors_for_many(phe_list, self.hpo_dict)
		q_hpo_ances_mat = get_csr_matrix_from_dict({0:item_list_to_rank_list(phe_extend_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		miss_log_prob = (self.dis_hpo_mat - self.dis_hpo_mat.multiply(q_hpo_ances_mat)).multiply(self.missing_rate_mat).sum(axis=1)  # np.matrix; shape=(dis_num, 1)
		score_vec = miss_log_prob.getA1()
		return score_vec


if __name__ == '__main__':
	pass