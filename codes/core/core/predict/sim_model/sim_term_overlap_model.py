

import os
import numpy as np
from tqdm import tqdm
import random
import heapq
from scipy.sparse import csr_matrix, save_npz, load_npz

from core.predict.model import SparseVecModel, Model
from core.utils.utils import get_all_ancestors_for_many, get_all_descendents_for_many, item_list_to_rank_list
from core.utils.utils import ret_same, get_csr_matrix_from_dict, delete_redundacy, delete_redundacy_for_mat, slice_list_with_keep_set
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_DESCENDENT, PHELIST_REDUCE, MODEL_PATH, TRAIN_MODE, PREDICT_MODE
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.ic_calculator import get_hpo_IC_dict, get_hpo_IC_vec

# ======================================================================================================================
# ======================================================================================================================
class SimTOModel(SparseVecModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, init_para=True):
		"""Term Overlap
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(SimTOModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'SimTOModel' # 'SimTOModel';
		self.init_save_path()
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		self.cal_dis_vec_mat()


	def cal_dis_vec_mat(self):
		row, col, data = [], [], []
		for i in range(self.DIS_CODE_NUMBER):  # ith disease
			hpo_list = self.dis2hpo[self.dis_list[i]]
			rank_list = list(map(lambda hpo_code: self.hpo_map_rank[hpo_code], hpo_list))
			row.extend([i]*len(rank_list))
			col.extend(rank_list)
			data.extend([1]*len(rank_list))
		self.dis_vec_mat = csr_matrix((data, (row, col)), shape=(self.DIS_CODE_NUMBER, self.HPO_CODE_NUMBER))


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return self.cal_score_for_phe_matrix(phe_matrix)


	def cal_score_for_phe_matrix(self, phe_matrix):
		return self.cal_dot_product(phe_matrix)


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			scipy.sparse.csr.csr_matrix: shape=[1, hpo_num]
		"""
		rank_list = item_list_to_rank_list(phe_list, self.hpo_map_rank)
		m = csr_matrix(([1]*len(rank_list), ([0]*len(rank_list), rank_list)), shape=(1, self.HPO_CODE_NUMBER))
		return m


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'SimTOModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_VEC_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_vec_mat.npz')


	def save(self):
		save_npz(self.DIS_VEC_MAT_NPZ, self.dis_vec_mat)


	def load(self):
		self.dis_vec_mat = load_npz(self.DIS_VEC_MAT_NPZ)


# ======================================================================================================================
# ======================================================================================================================
class SimTOQReduceMaImSpModel(Model):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(SimTOQReduceMaImSpModel, self).__init__()
		self.name = 'SimTOQReduceMaImSpModel'  # Match+Impre+Specified
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_hpo_dict()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()
		self.dis_list = hpo_reader.get_dis_list()
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.HPO_NUM = hpo_reader.get_hpo_num()
		self.dis_hpo_ances_mat = None  # csr_matrix; shape=[dis_num, hpo_num]
		self.dis_hpo_descen_mat = None # csr_matrix; shape=[dis_num, hpo_num]
		if init_para:
			self.train()


	def train(self):
		self.dis_hpo_ances_mat = get_csr_matrix_from_dict(
			self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR), shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True
		)
		self.dis_hpo_descen_mat = get_csr_matrix_from_dict(
			self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_DESCENDENT), shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True
		)

	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_descen_mat.multiply(q_hpo_mat)    # csr_matrix; shape=[dis_num, hpo_num]
		score_vec = match_mat.sum(axis=1).getA1()  # np.array; shape=dis_num
		return score_vec


	def query(self, phe_list, topk=10):
		score_vec = self.cal_score(phe_list)
		assert np.sum(np.isnan(score_vec)) == 0  #
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1])   # [(dis_code, score), ...], shape=(dis_num, )


# ======================================================================================================================
class SimTODQAcrossModel(Model):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(SimTODQAcrossModel, self).__init__()
		self.name = 'SimTODQAcrossModel'  # Match+Impre+Specified
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_hpo_dict()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()
		self.dis_list = hpo_reader.get_dis_list()
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.HPO_NUM = hpo_reader.get_hpo_num()
		self.dis_hpo_mat = None # csr_matrix; shape=[dis_num, hpo_num]
		self.dis_hpo_ances_mat = None  # csr_matrix; shape=[dis_num, hpo_num]
		if init_para:
			self.train()


	def train(self):
		self.dis_hpo_mat = get_csr_matrix_from_dict(
			self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_REDUCE), shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True
		)
		self.dis_hpo_ances_mat = get_csr_matrix_from_dict(
			self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR), shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True
		)


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		phe_ances_list = get_all_ancestors_for_many(phe_list, self.hpo_dict)
		q_hpo_ances_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_ances_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_mat.multiply(q_hpo_ances_mat)    # csr_matrix; shape=[dis_num, hpo_num]
		score_vec = match_mat.sum(axis=1).getA1()  # np.array; shape=dis_num
		return score_vec


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		score_vec = self.cal_score(phe_list)  # shape=[dis_num]
		assert np.sum(np.isnan(score_vec)) == 0  #
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


	def query(self, phe_list, topk=10):
		"""
		Args:
			phe_list (list): list of phenotype
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		if len(phe_list) == 0:
			return self.query_empty(topk)
		score_vec = self.query_score_vec(phe_list)
		return self.score_vec_to_result(score_vec, topk)


# ======================================================================================================================
class SimTODominantRandomDQAcrossModel(SimTODQAcrossModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantRandomDQAcrossModel, self).__init__(hpo_reader, init_para=False)
		self.name = 'SimTODominantRandomDQAcrossModel'
		if init_para:
			self.train()

	def score_item_to_score(self, score_item):
		return score_item[0]

	def cal_score(self, phe_list):
		score_vec = super(SimTODominantRandomDQAcrossModel, self).cal_score(phe_list)
		return list(zip(score_vec, [random.random() for _ in range(len(score_vec))]))

	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantRandomDQAcrossModel, self).query(phe_list, topk)
		# print(dis_score_item_list)
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in dis_score_item_list]


# ======================================================================================================================
class SimTODominantRandomQReduceMaImSpModel(SimTOQReduceMaImSpModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantRandomQReduceMaImSpModel, self).__init__(hpo_reader, init_para=False)
		self.name = 'SimTODominantRandomQReduceMaImSpModel'
		if init_para:
			self.train()

	def score_item_to_score(self, score_item):
		return score_item[0]

	def cal_score(self, phe_list):
		score_vec = super(SimTODominantRandomQReduceMaImSpModel, self).cal_score(phe_list)
		return list(zip(score_vec, [random.random() for _ in range(len(score_vec))]))

	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantRandomQReduceMaImSpModel, self).query(phe_list, topk)
		print(dis_score_item_list)
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in dis_score_item_list]


# ======================================================================================================================
class SimTOQReduceModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTOQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = 'SimTOQReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class SimTODReduceModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, init_para=False)
		self.name = 'SimTODReduceModel' # SimICTOQReduceModel
		if init_para:
			self.train()


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
# ======================================================================================================================
class ICTOModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, model_name=None, init_para=True):
		super(ICTOModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = model_name or 'ICTOModel'
		if init_para:
			self.train()


	def train(self):
		super(ICTOModel, self).train()
		IC_dict = get_hpo_IC_dict(self.hpo_reader) # {HPO_CODE: IC, ...}
		self.IC_vec = np.array([IC_dict[hpo_code] for hpo_code in self.hpo_list])   # np.array; shape = [HPONum]


	def cal_score(self, phe_list):
		phe_matrix = self.phe_list_to_matrix(phe_list)
		phe_matrix = phe_matrix.multiply(self.IC_vec)
		return self.cal_dot_product(phe_matrix)


# ======================================================================================================================
class ICTOQReduceModel(ICTOModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(ICTOQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = 'ICTOQReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class ICTOQReduceMaImSpModel(SimTOQReduceMaImSpModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(ICTOQReduceMaImSpModel, self).__init__(hpo_reader, init_para=False)
		self.name = 'ICTOQReduceMaImSpModel'
		if init_para:
			self.train()


	def train(self):
		super(ICTOQReduceMaImSpModel, self).train()
		IC_dict = get_hpo_IC_dict(self.hpo_reader, default_IC=0.0) # {HPO_CODE: IC, ...}
		self.IC_vec = np.array([IC_dict[hpo_code] for hpo_code in self.hpo_reader.get_hpo_list()])   # np.array; shape = [HPONum]


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_descen_mat.multiply(q_hpo_mat)    # csr_matrix; shape=[dis_num, hpo_num]
		score_vec = match_mat.multiply(self.IC_vec).sum(axis=1).getA1()  # np.array; shape=dis_num
		return score_vec


# ======================================================================================================================
class ICTODQAcrossModel(SimTODQAcrossModel):
	def __init__(self, hpo_reader=HPOReader(), sym_mode='union', mode=TRAIN_MODE, model_name=None, init_para=True, slice_no_anno=False):
		super(ICTODQAcrossModel, self).__init__(hpo_reader, init_para=False)
		self.init_save_path()
		self.name = 'ICTODQAcrossModel' if model_name is None else model_name
		self.sym_mode = sym_mode
		self.anno_hpo_set = set(self.hpo_reader.get_anno_hpo_list()) if slice_no_anno else None
		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		super(ICTODQAcrossModel, self).train()
		IC_dict = get_hpo_IC_dict(self.hpo_reader, default_IC=0.0) # {HPO_CODE: IC, ...}
		self.IC_vec = np.array([IC_dict[hpo_code] for hpo_code in self.hpo_reader.get_hpo_list()])   # np.array; shape = [HPONum]


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		phe_list = super(ICTODQAcrossModel, self).process_query_phe_list(phe_list, phe_list_mode, hpo_dict)
		if self.anno_hpo_set is not None:
			phe_list = slice_list_with_keep_set(phe_list, self.anno_hpo_set)
		return phe_list


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		phe_ances_list = get_all_ancestors_for_many(phe_list, self.hpo_dict)
		q_hpo_ances_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_ances_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)

		if self.sym_mode == 'union':
			match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_mat.multiply(q_hpo_ances_mat)    # csr_matrix; shape=[dis_num, hpo_num]
			return match_mat.multiply(self.IC_vec).sum(axis=1).getA1()  # np.array; shape=dis_num
		elif self.sym_mode == 'ave':
			return self.dis_hpo_ances_mat.multiply(q_hpo_mat).multiply(self.IC_vec).sum(
				axis=1).getA1() + self.dis_hpo_mat.multiply(q_hpo_ances_mat).multiply(self.IC_vec).sum(axis=1).getA1()
		else:
			assert False


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'ICTODQAcrossModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.IC_VEC_NPY = os.path.join(self.SAVE_FOLDER, 'IC_vec.npy')
		self.DIS_HPO_ANCES_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_hpo_ances_mat.npz')
		self.DIS_HPO_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_hpo_mat.npz')


	def save(self):
		np.save(self.IC_VEC_NPY, self.IC_vec)
		save_npz(self.DIS_HPO_ANCES_MAT_NPZ, self.dis_hpo_ances_mat)
		save_npz(self.DIS_HPO_MAT_NPZ, self.dis_hpo_mat)


	def load(self):
		self.IC_vec = np.load(self.IC_VEC_NPY)
		self.dis_hpo_ances_mat = load_npz(self.DIS_HPO_ANCES_MAT_NPZ)
		self.dis_hpo_mat = load_npz(self.DIS_HPO_MAT_NPZ)


class ICTODQAcrossModel2(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), model_name=None, init_para=True):
		super(ICTODQAcrossModel2, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = model_name or 'ICTODQAcrossModel2'
		if init_para:
			self.train()


	def train(self):
		super(ICTODQAcrossModel2, self).train()
		self.hpo_ances_mat = self.hpo_reader.get_hpo_ances_mat(contain_self=False)
		self.IC_vec = get_hpo_IC_vec(self.hpo_reader)


	def cal_score_for_phe_matrix(self, phe_matrix):
		m = self.dis_vec_mat.multiply(phe_matrix)
		m = delete_redundacy_for_mat(m, self.hpo_ances_mat)


		return m.multiply(self.IC_vec).sum(axis=1).getA1()


# ======================================================================================================================
class ICTODReduceModel(ICTOModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(ICTODReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, init_para=True)
		self.name = 'ICTODReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
# ======================================================================================================================
class ProbTOModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, default_prob=0.9, init_para=True):

		super(ProbTOModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = 'ProbTOModelDP{0}'.format(default_prob)
		self.default_prob=default_prob
		if init_para:
			self.train()


	def train(self):
		super(ProbTOModel, self).train()
		dis_map_rank, hpo_map_rank = self.hpo_reader.get_dis_map_rank(), self.hpo_reader.get_hpo_map_rank()
		hpo_dict = self.hpo_reader.get_hpo_dict()
		dis_to_hpo_prob = self.hpo_reader.get_dis_to_hpo_prob(default_prob=self.default_prob)    # {dis_code: [[hpo_code, prob], ...]}
		row, col, data = [], [], []
		for dis_code, hpo_prob_list in tqdm(dis_to_hpo_prob.items()):
			hpo_prob_dict = {hpo_code: prob for hpo_code, prob in hpo_prob_list}
			self.extend_hpo_prob_dict(hpo_prob_dict, hpo_dict)
			row.extend([dis_map_rank[dis_code]]*len(hpo_prob_dict))
			hpo_list, prob_list = zip(*hpo_prob_dict.items())
			col.extend(item_list_to_rank_list(hpo_list, hpo_map_rank))
			data.extend(prob_list)
		data = np.log(data)
		self.prob_mat = csr_matrix((data, (row, col)), shape=(self.DIS_CODE_NUMBER, self.HPO_CODE_NUMBER), dtype=np.float32)


	def extend_hpo_prob_dict(self, hpo_prob_dict, hpo_dict):

		ancestor_set = get_all_ancestors_for_many(list(hpo_prob_dict.keys()), hpo_dict)
		self.get_hpo_prob('HP:0000001', hpo_prob_dict, hpo_dict, ancestor_set)


	def get_hpo_prob(self, hpo_code, hpo_prob_dict, hpo_dict, ancestor_set):
		if hpo_code not in ancestor_set:
			return -1
		if hpo_code in hpo_prob_dict:
			return hpo_prob_dict[hpo_code]
		prob = max([self.get_hpo_prob(childCode, hpo_prob_dict, hpo_dict, ancestor_set) for childCode in hpo_dict[hpo_code].get('CHILD', [])])
		hpo_prob_dict[hpo_code] = prob
		assert prob > 0
		return prob


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		match_mat = self.dis_vec_mat.multiply(phe_matrix)
		score_vec = match_mat.multiply(self.prob_mat).sum(axis=1) / match_mat.sum(axis=1)
		score_vec[np.isnan(score_vec)] = -np.inf
		return score_vec.getA1()


# ======================================================================================================================
class ProbTOQReduceModel(ProbTOModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, init_para=True):

		super(ProbTOQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, default_prob, init_para=False)
		self.name = 'ProbTOQReduceModelDP{0}'.format(default_prob)
		if init_para:
			self.train()


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class ProbTODReduceModel(ProbTOModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, init_para=True):

		super(ProbTODReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, default_prob, init_para=False)
		self.name = 'ProbTODReduceModelDP{0}'.format(default_prob)
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
# ======================================================================================================================
class SimTODominantICModel(ICTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, init_para=True):
		super(SimTODominantICModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = 'SimTODominantICModel'
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]


	def cal_score(self, phe_list):
		phe_matrix = self.phe_list_to_matrix(phe_list)
		match_mat = self.dis_vec_mat.multiply(phe_matrix)
		match_num_vec = match_mat.sum(axis=1).getA1()  # shape=[dis_num, 1]
		IC_score_vec = match_mat.multiply(self.IC_vec).sum(axis=1).getA1()  # shape=[dis_num, 1]
		# IC_score_vec = match_mat.multiply(self.IC_vec).max(axis=1).toarray().flatten()  # shape=[dis_num, 1] # try
		return list(zip(match_num_vec, IC_score_vec))


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			ret = sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]), reverse=True)
		else:
			ret = heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]))  # [(dis_code, score), ...], shape=(dis_num, )
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in ret]


	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantICModel, self).query(phe_list, topk)

		return dis_score_item_list


# ======================================================================================================================
class SimTOQReduceDominantICModel(SimTODominantICModel):
	def __init__(self, hpo_reader=HPOReader(), model_name=None, init_para=True):
		super(SimTOQReduceDominantICModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = model_name or 'SimTOQReduceDominantICModel'
		if init_para:
			self.train()


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class SimTODominantICDReduceModel(SimTODominantICModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantICDReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, init_para=False)
		self.name = 'SimTODominantICDReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
class SimTODominantICQReduceMaImSpModel(ICTOQReduceMaImSpModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantICQReduceMaImSpModel, self).__init__(hpo_reader, init_para=False)
		self.name = 'SimTODominantICQReduceMaImSpModel'
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_descen_mat.multiply(q_hpo_mat)    # csr_matrix; shape=[dis_num, hpo_num]
		match_num_vec = match_mat.sum(axis=1)   # shape=[dis_num, 1]
		IC_score_vec = match_mat.multiply(self.IC_vec).sum(axis=1)   # shape=[dis_num, 1]
		return list(zip(match_num_vec.getA1(), IC_score_vec.getA1()))


	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantICQReduceMaImSpModel, self).query(phe_list, topk)

		return dis_score_item_list


# ======================================================================================================================
class SimTODominantICDQAcrossModel(ICTODQAcrossModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantICDQAcrossModel, self).__init__(hpo_reader, init_para=False)
		self.name = 'SimTODominantICDQAcrossModel'
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]


	def cal_score(self, phe_list):
		phe_list = delete_redundacy(phe_list, self.hpo_dict)
		q_hpo_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		phe_ances_list = get_all_ancestors_for_many(phe_list, self.hpo_dict)
		q_hpo_ances_mat = get_csr_matrix_from_dict({0: item_list_to_rank_list(phe_ances_list, self.hpo_map_rank)}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		match_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat) + self.dis_hpo_mat.multiply(q_hpo_ances_mat)    # csr_matrix; shape=[dis_num, hpo_num]
		match_num_vec = match_mat.sum(axis=1)   # shape=[dis_num, 1]
		IC_score_vec = match_mat.multiply(self.IC_vec).sum(axis=1)   # shape=[dis_num, 1]
		return list(zip(match_num_vec.getA1(), IC_score_vec.getA1()))


	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantICDQAcrossModel, self).query(phe_list, topk)

		return dis_score_item_list


# ======================================================================================================================
# ======================================================================================================================
class SimTODominateProbModel(ProbTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, default_prob=0.9, init_para=True):
		super(SimTODominateProbModel, self).__init__(hpo_reader, phe_list_mode, default_prob, init_para=False)
		self.name = 'SimTODominateProbModelDP{0}'.format(default_prob)
		if init_para:
			self.train()


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			np.array or list: score_itemVec, [score_item1, score_item2], shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		match_mat = self.dis_vec_mat.multiply(phe_matrix)
		match_num_vec = match_mat.sum(axis=1)  # shape=[dis_num, 1]
		prob_score_vec = match_mat.multiply(self.prob_mat).sum(axis=1) / match_num_vec
		prob_score_vec[np.isnan(prob_score_vec)] = -np.inf
		return list(zip(match_num_vec.getA1(), prob_score_vec.getA1()))


	def score_item_vec_to_score_vec(self, score_itemVec):
		ret_list = [score_item[0] for score_item in len(score_itemVec)]
		if isinstance(score_itemVec, np.ndarray):
			ret_list = np.array(ret_list)
		return ret_list


	def score_item_to_score(self, score_item):
		return score_item[0]


	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominateProbModel, self).query(phe_list, topk)

		return dis_score_item_list


# ======================================================================================================================
class SimTODominateProbQReduceModel(SimTODominateProbModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, init_para=True):

		super(SimTODominateProbQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, default_prob, init_para=False)
		self.name = 'SimTODominateProbQReduceModelDP{0}'.format(default_prob)
		if init_para:
			self.train()


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class SimTODominateProbDReduceModel(SimTODominateProbModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.9, init_para=True):

		super(SimTODominateProbDReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, default_prob, init_para=False)
		self.name = 'SimTODominateProbDReduceModelDP{0}'.format(default_prob)
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
# ======================================================================================================================
class SimTODominantRandomModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, init_para=True):
		super(SimTODominantRandomModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = 'SimTODominantRandomModel'
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]


	def cal_score(self, phe_list):
		score_vec = super(SimTODominantRandomModel, self).cal_score(phe_list)
		return list(zip(score_vec, [random.random() for _ in range(len(score_vec))]))


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			ret = sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]), reverse=True)
		else:
			ret = heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:tuple(item[1]))  # [(dis_code, score), ...], shape=(dis_num, )
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in ret]


# ======================================================================================================================
class SimTODominantRandomQReduceModel(SimTODominantRandomModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantRandomQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = 'SimTODominantRandomQReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class SimTODominantRandomDReduceModel(SimTODominantRandomModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantRandomDReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, init_para=False)
		self.name = 'SimTODominantRandomDReduceModel'
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


# ======================================================================================================================
# ======================================================================================================================
class SimTODominantReverseModel(SimTOModel):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, init_para=True):
		super(SimTODominantReverseModel, self).__init__(hpo_reader, phe_list_mode, init_para=False)
		self.name = 'SimTODominantNormalModel' # SimTODominantNormalModel
		if init_para:
			self.train()


	def score_item_to_score(self, score_item):
		return score_item[0]



	def cal_score(self, phe_list):  # dominantNormal
		score_vec = super(SimTODominantReverseModel, self).cal_score(phe_list)
		second = list(range(len(score_vec))); second.reverse()
		return list(zip(score_vec, second))


	def query(self, phe_list, topk=10):
		dis_score_item_list = super(SimTODominantReverseModel, self).query(phe_list, topk)
		# print(dis_score_item_list)
		return [(dis_code, self.score_item_to_score(score_item)) for dis_code, score_item in dis_score_item_list]


# ======================================================================================================================
class SimTODominantReverseQReduceModel(SimTODominantReverseModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantReverseQReduceModel, self).__init__(hpo_reader, PHELIST_ANCESTOR, init_para=False)
		self.name = 'SimTODominantReverseQReduceModel'  # SimTODominantNormalQReduceModel
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return delete_redundacy(phe_list, hpo_dict)


# ======================================================================================================================
class SimTODominantReverseDReduceModel(SimTODominantReverseModel):
	def __init__(self, hpo_reader=HPOReader(), init_para=True):
		super(SimTODominantReverseDReduceModel, self).__init__(hpo_reader, PHELIST_REDUCE, init_para=False)
		self.name = 'SimTODominantReverseDReduceModel'  # SimTODominantNormalDReduceModel
		if init_para:
			self.train()

	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		return list(get_all_ancestors_for_many(phe_list, hpo_dict))


if __name__ == '__main__':
	# from core.predict.PageRankNoiseReductor import PageRankNoiseReductor
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()



	model = ICTODQAcrossModel(hpo_reader)
