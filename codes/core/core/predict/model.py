import shutil
import os
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import heapq
from scipy.sparse import vstack
from sklearn.metrics import f1_score, accuracy_score
import joblib
import scipy
import random
import pyemd
from copy import deepcopy

from core.utils.constant import SET_SIM_ASYMMAX_QD, SET_SIM_ASYMMAX_DQ, SET_SIM_SYMMAX, SET_SIM_EMD, SEED
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, VEC_TYPE_0_1_DIM_REDUCT, VEC_TYPE_TF, VEC_TYPE_TF_IDF, VEC_TYPE_IDF
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_ORIGIN, PHELIST_REDUCE, PHELIST_ANCESTOR_DUP, VEC_COMBINE_MAX, VEC_COMBINE_MEAN, VEC_COMBINE_SUM
from core.utils.utils import item_list_to_rank_list, mat_l2_norm, data_to_01_matrix, data_to_01_dense_matrix, data_to_tf_matrix
from core.utils.utils import cal_macro_auc, get_all_ancestors_for_many, delete_redundacy, get_all_dup_ancestors_for_many, check_return
from core.utils.utils import slice_list_with_keep_set
from core.reader import HPOReader, RDFilterReader
from core.helper.data.data_helper import DataHelper
from core.predict.config import Config


class BaseModel(object):
	def __init__(self):
		pass


class Model(BaseModel):
	def __init__(self):
		super(Model, self).__init__()
		self.dhelper = None


	def query_score_vec(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo_code1, ...]
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		raise NotImplementedError


	def score_vec_to_result(self, score_vec, topk):
		"""
		Args:
			score_vec (np.ndarray): (dis_num,)
			topk (int or None):
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		raise NotImplementedError


	def query(self, phe_list, topk):
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


	def get_data_helper(self):
		if self.dhelper is None:
			self.dhelper = DataHelper(self.hpo_reader) if hasattr(self, 'hpo_reader') else DataHelper()
		return self.dhelper


	def query_score_mat(self, phe_lists, chunk_size=200, cpu_use=12):
		"""
		Returns:
			np.ndarray: shape=(sample_num, dis_num)
		"""
		para_list = phe_lists
		if cpu_use == 1 or cpu_use is None:
			return np.vstack([self.query_score_vec(paras) for paras in para_list])
		with Pool(cpu_use) as pool:
			return np.array([result for result in tqdm(pool.imap(self.query_score_vec, para_list, chunksize=chunk_size), total=len(para_list), leave=False)])


	def score_mat_to_results(self, phe_lists, score_mat, topk):
		return [
			self.score_vec_to_result(score_mat[i], topk) if len(phe_lists[i]) != 0 else self.query_empty(topk)
			for i in range(score_mat.shape[0])
		]


	def query_many(self, phe_lists, topk=10, chunk_size=200, cpu_use=12):
		"""
		Args:
			phe_lists (list): [[hpo1, hpo2, ...], ...]
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list: [result1, result2, ...], result=[(dis1, score1), ...], scores decreasing
		"""
		score_mat = self.query_score_mat(phe_lists, chunk_size, cpu_use)
		return self.score_mat_to_results(phe_lists, score_mat, topk)


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		if phe_list_mode == PHELIST_ANCESTOR:
			return list(get_all_ancestors_for_many(phe_list, hpo_dict))
		elif phe_list_mode == PHELIST_REDUCE:
			return delete_redundacy(phe_list, hpo_dict)
		elif PHELIST_ANCESTOR_DUP:
			return get_all_dup_ancestors_for_many(phe_list, hpo_dict)
		return phe_list


	def query_empty(self, topk):
		if not hasattr(self, 'hpo_reader'):
			self.hpo_reader = HPOReader()
		dis_list = deepcopy(self.hpo_reader.get_dis_list())
		random.shuffle(dis_list)
		dis_score_list = [(dis_code, 0.0) for dis_code in dis_list]
		if topk == None:
			return dis_score_list
		return dis_score_list[:topk]


	def query_empty_score_vec(self):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		dis_num = self.hpo_reader.get_dis_num()
		return np.random.rand(dis_num)


	def explain(self, *args, **kwargs):
		return ''


	def explain_as_str(self, *args, **kwargs):
		return ''


class ScoreMatModel(Model):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ORIGIN, set_sim_method=SET_SIM_SYMMAX):
		super(ScoreMatModel, self).__init__()
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_slice_hpo_dict()
		self.hpo_list = hpo_reader.get_hpo_list()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()
		self.HPO_CODE_NUMBER = len(self.hpo_list)
		self.phe_list_mode = phe_list_mode
		self.dis_list = hpo_reader.get_dis_list()
		self.dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(phe_list_mode)
		self.DIS_CODE_NUMBER = hpo_reader.get_dis_num()

		self.set_sim_method = set_sim_method
		if set_sim_method == SET_SIM_SYMMAX:
			self.score_mat = None    # shape=[len(hpo_dict), len(hpo_dict)]
			self.phe_set_qd_sim = self.phe_set_sym_sim
		elif set_sim_method == SET_SIM_ASYMMAX_QD:
			self.score_mat = None  # shape=[len(hpo_dict), len(hpo_dict)]
			self.phe_set_qd_sim = self.phe_set_q_to_d_asym_sim
		elif set_sim_method == SET_SIM_ASYMMAX_DQ:
			self.score_mat = None  # shape=[len(hpo_dict), len(hpo_dict)]
			self.phe_set_qd_sim = self.phe_set_d_to_q_asym_sim
		elif set_sim_method == SET_SIM_EMD:
			self.distance_mat = None #
			self.phe_set_qd_sim = self.phe_set_sim_emd
		else:
			assert False


	def cal_score_mat(self):
		raise NotImplementedError


	def train(self):
		print('training...')
		if self.set_sim_method == SET_SIM_SYMMAX or self.set_sim_method == SET_SIM_ASYMMAX_QD or self.set_sim_method == SET_SIM_ASYMMAX_DQ:
			self.cal_score_mat()
		else:
			self.cal_distance_mat()
		print('training end.')


	def get_score_mat(self):
		if self.score_mat is None:
			self.cal_score_mat()
		return self.score_mat


	def cal_distance_mat(self):
		self.cal_score_mat()
		self.distance_mat = -self.score_mat
		del self.score_mat


	def phe_set_sim_base(self, m):

		return np.mean(m[list(range(len(m))), np.argmax(m, axis=1)])


	def phe_set_asym_sim(self, hpoRankList1, hpoRankList2):
		"""set1 -> set2
		"""
		m = self.score_mat[hpoRankList1, :][:, hpoRankList2]
		return self.phe_set_sim_base(m)


	def phe_set_sym_sim(self, hpoRankList1, hpoRankList2):
		"""set1 <-> set2
		Args:
			hpoRankList1 (list): [rank1, rank2, ...], type(rank)=int
			hpoRankList2 (list): [rank1, rank2, ...], type(rank)=int
		Returns:
			float: similarity
		"""
		m = self.score_mat[hpoRankList1, :][:, hpoRankList2]
		score12 = self.phe_set_sim_base(m)
		score21 = self.phe_set_sim_base(m.T)
		return (score12 + score21) / 2


	def phe_set_q_to_d_asym_sim(self, QHPORankList, DHPORankList):
		return self.phe_set_asym_sim(QHPORankList, DHPORankList)


	def phe_set_d_to_q_asym_sim(self, QHPORankList, DHPORankList):
		return self.phe_set_asym_sim(DHPORankList, QHPORankList)


	def phe_set_sim_emd(self, hpoRankList1, hpoRankList2):
		cols = np.unique(hpoRankList1 + hpoRankList2)
		hpo_rank_map_col = {cols[i]: i for i in range(len(cols))}
		pq = np.zeros(shape=(2, len(cols)), dtype=np.float64)
		for hpo_rank in hpoRankList1:
			pq[0, hpo_rank_map_col[hpo_rank]] = 1
		for hpo_rank in hpoRankList2:
			pq[1, hpo_rank_map_col[hpo_rank]] = 1
		m = self.distance_mat[:, cols][cols, :]
		return -pyemd.emd(pq[0], pq[1], m, extra_mass_penalty=0)


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_list = item_list_to_rank_list(phe_list, self.hpo_map_rank)
		return np.array([self.phe_set_qd_sim(phe_list, self.dis_int_to_hpo_int[i]) for i in range(self.DIS_CODE_NUMBER)])


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		score_vec = self.cal_score(phe_list)  # shape=[dis_num]
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


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




class SparseVecModel(Model):
	def __init__(self, hpo_reader, phe_list_mode=PHELIST_ORIGIN):
		super(SparseVecModel, self).__init__()
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_hpo_dict()
		self.hpo_list = hpo_reader.get_hpo_list()    # [hpo_code1, ...]
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()    # {rank: hpo_code, ...}
		self.HPO_CODE_NUMBER = len(self.hpo_list)
		self.phe_list_mode = phe_list_mode
		self.dis2hpo = hpo_reader.get_dis_to_hpo_dict(phe_list_mode)  # {dis_code: [hpo_code, ...]}
		self.dis_list = hpo_reader.get_dis_list()    # [disease_code1, ...]
		self.DIS_CODE_NUMBER = len(self.dis_list)
		self.dis_vec_mat = None   # shape=[dis_num, len(disVec)], type=scipy.sparse.csr.csr_matrix


	def cal_dis_vec_mat(self):
		raise NotImplementedError


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		raise NotImplementedError


	def mat_l2_norm(self, m):
		"""
		Args:
			m (scipy.sparse.csr.csr_matrix)
		"""
		return m.multiply(1/np.sqrt(m.power(2).sum(axis=1)))


	def cal_dot_product(self, phe_matrix):
		"""
		Args:
			phe_matrix (scipy.sparse.csr.csr_matrix): shape=[1, len(disVec)]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		m = self.dis_vec_mat.dot(phe_matrix.transpose())    # [dis_num, 1]
		return m.toarray().flatten()    # [dis_num]


	def cal_euclid_dist(self, phe_matrix):
		"""
		Args:
			phe_matrix (scipy.sparse.csr.csr_matrix): shape=[1, len(disVec)]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		return np.sqrt(self.cal_euclid_dist2(phe_matrix))  # [dis_num]


	def cal_euclid_dist2(self, phe_matrix):
		m = (self.dis_vec_mat - vstack([phe_matrix] * self.dis_vec_mat.shape[0])).power(2).sum(axis=1)  # numpy.matrixlib.defmatrix.matrix
		return m.A1  # [dis_num]


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		score_vec = self.cal_score(phe_list)  # shape=[dis_num]
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


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


class DenseVecModel(Model):
	def __init__(self, hpo_reader, phe_list_mode=PHELIST_ORIGIN):
		super(DenseVecModel, self).__init__()
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_hpo_dict()
		self.hpo_list = hpo_reader.get_hpo_list()    # [hpo_code1, ...]
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()    # {rank: hpo_code, ...}
		self.HPO_CODE_NUMBER = hpo_reader.get_hpo_num()
		self.phe_list_mode = PHELIST_ORIGIN
		self.dis2hpo = hpo_reader.get_dis_to_hpo_dict(phe_list_mode)
		self.dis_list = hpo_reader.get_dis_list()    # [disease_code1, ...]
		self.DIS_CODE_NUMBER = hpo_reader.get_dis_num()
		self.dis_vec_mat = None   # shape=[dis_num, len(disVec)], type=scipy.sparse.csr.csr_matrix


	def cal_dis_vec_mat(self):
		raise NotImplementedError


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		raise NotImplementedError


	def mat_l2_norm(self, m):

		return mat_l2_norm(m)


	def cal_dot_product(self, phe_matrix):
		"""
		Args:
			phe_matrix (np.ndarray): shape=[1, len(disVec)]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		return self.dis_vec_mat.dot(phe_matrix.T).flatten()


	def cal_euclid_dist(self, phe_matrix):
		"""
		Args:
			phe_matrix (np.ndarray): shape=[1, len(disVec)]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		return np.sqrt(np.sum(np.power(self.dis_vec_mat-phe_matrix, 2), axis=1))


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		score_vec = self.cal_score(phe_list)  # shape=[dis_num]
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1])   # [(dis_code, score), ...], shape=(dis_num, )


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


class ClassificationModel(Model):
	def __init__(self, hpo_reader, vec_type, phe_list_mode=PHELIST_ORIGIN, embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,),
			dim_reductor=None, dp=1.0, up_induced_rule='max', use_rd_mix_code=False):
		"""
		Args:
			vec_type (str): VEC_TYPE_0_1 |
			embed_mat (np.ndarray): shape=(hpo_num, embed_size)
		"""
		super(ClassificationModel, self).__init__()
		self.hpo_reader = hpo_reader
		self.hpo_dict = hpo_reader.get_hpo_dict()
		self.hpo_list = hpo_reader.get_hpo_list()    # [hpo_code1, ...]
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()    # {hpo_code: rank, ...}
		self.HPO_CODE_NUMBER = hpo_reader.get_hpo_num()

		self.use_rd_mix_code = use_rd_mix_code
		if use_rd_mix_code:
			source_code_set = set(hpo_reader.get_dis_list())
			rd_reader = RDFilterReader(keep_source_codes=source_code_set)
			self.dis_list = rd_reader.get_rd_list()
			rd_to_sources = rd_reader.get_rd_to_sources()  # {RD_CODE: [SOURCE_CODE, ...]}
			self.rd_to_sources = {rd_code: slice_list_with_keep_set(source_codes, source_code_set) for rd_code, source_codes in rd_to_sources.items()}
			self.DIS_CODE_NUMBER = rd_reader.get_rd_num()
		else:
			self.dis_list = hpo_reader.get_dis_list()    # [disease_code1, ...]
			self.DIS_CODE_NUMBER = hpo_reader.get_dis_num()

		self.phe_list_mode = phe_list_mode
		self.dp = dp
		self.up_induce_rule = up_induced_rule

		self.vec_type = vec_type
		if self.vec_type == VEC_TYPE_0_1:
			self.raw_X_to_X_func = self.raw_X_to_01_X
		elif self.vec_type == VEC_TYPE_EMBEDDING:
			self.raw_X_to_X_func = self.raw_X_to_embed_X
			self.hpo_embed_mat = embed_mat # np.ndarray; shape=(hpo_num, embed_size)
			self.embed_size = embed_mat.shape[1]
			self.combine_modes = combine_modes
			self.combine_func_dict = {VEC_COMBINE_MEAN: self.raw_X_to_mean_embed_X, VEC_COMBINE_SUM: self.raw_X_to_sum_embed_X, VEC_COMBINE_MAX: self.raw_X_to_max_embed_X}
			assert embed_mat.shape[0] == self.HPO_CODE_NUMBER
		elif self.vec_type == VEC_TYPE_0_1_DIM_REDUCT:
			self.raw_X_to_X_func = self.raw_X_to_reduct_X
			self.dr = dim_reductor
		elif self.vec_type == VEC_TYPE_TF:
			self.raw_X_to_X_func = self.raw_X_to_tf_X
		elif self.vec_type == VEC_TYPE_TF_IDF:
			self.raw_X_to_X_func = self.raw_X_to_tf_idf_X
		elif self.vec_type == VEC_TYPE_IDF:
			self.raw_X_to_X_func = self.raw_X_to_idf_X
		else:
			self.raw_X_to_X_func = self.raw_X_to_X # subclass defined


	def raw_X_to_X(self, raw_X):
		raise NotImplementedError


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		raise NotImplementedError


	def before_phe_lists_to_X(self, phe_lists):
		if type(phe_lists) == set:
			phe_lists = list(phe_lists)
		if type(phe_lists[0]) == str:
			phe_lists = [phe_lists]
		new_phe_list = []
		for phe_list in phe_lists:  # slice dict
			new_phe_list.append([hpo for hpo in phe_list if hpo in self.hpo_map_rank])
		return phe_lists


	def raw_X_to_01_X(self, raw_X):

		return data_to_01_matrix(raw_X, self.HPO_CODE_NUMBER)


	def raw_X_to_tf_X(self, raw_X):
		return data_to_tf_matrix(raw_X, self.HPO_CODE_NUMBER)


	def raw_X_to_tf_idf_X(self, raw_X):
		return self.get_data_helper().data_to_tf_idf_matrix(raw_X, self.HPO_CODE_NUMBER)


	def raw_X_to_idf_X(self, raw_X):
		return self.get_data_helper().data_to_idf_matrix(raw_X, self.HPO_CODE_NUMBER)


	def raw_X_to_01_dense_X(self, raw_X):

		return data_to_01_dense_matrix(raw_X, self.HPO_CODE_NUMBER)


	def raw_X_to_reduct_X(self, raw_X):
		return self.dr.transform(self.raw_X_to_01_dense_X(raw_X))


	def raw_X_to_mean_embed_X(self, raw_X):
		return np.vstack([np.mean(self.hpo_embed_mat[hpo_int_list, :], axis=0) for hpo_int_list in raw_X])


	def raw_X_to_weight_mean_embed_X(self, raw_X):
		# FIXME
		pass


	def raw_X_to_sum_embed_X(self, raw_X):
		return np.vstack([np.sum(self.hpo_embed_mat[hpo_int_list, :], axis=0) for hpo_int_list in raw_X])


	def raw_X_to_max_embed_X(self, raw_X):
		return np.vstack([np.max(self.hpo_embed_mat[hpo_int_list, :], axis=0) for hpo_int_list in raw_X])


	def raw_X_to_embed_X(self, raw_X):
		return np.hstack([self.combine_func_dict[mode](raw_X) for mode in self.combine_modes])


	def raw_X_doing_nothing_to_X(self, raw_X):
		if isinstance(raw_X, np.ndarray):
			return raw_X
		return np.array(raw_X)


	def phe_lists_to_X(self, phe_lists):
		"""
		Args:
			phe_lists (list): [hpo1, hpo2] | [[hpo1, hpo2], [], ...]
		Returns:
			scipy.sparse.csr.csr_matrix: X, shape=[len(phe_lists), hpo_num]
		"""
		phe_lists = self.before_phe_lists_to_X(phe_lists)
		raw_X = [item_list_to_rank_list(sample, self.hpo_map_rank) for sample in phe_lists]
		return self.raw_X_to_X_func(raw_X)


	def eval(self, raw_X, y_, sw, logger, chunk_size=None):
		sample_size = len(raw_X)
		if chunk_size is None:
			chunk_size = sample_size
		y, pred_prob = [], []  # shape=(sample_num,); # shape=(sample_num, class_num)
		intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
		for i in range(len(intervals)-1):
			X = self.raw_X_to_X_func(raw_X[intervals[i]: intervals[i+1]])
			prob = self.predict_prob(X)
			pred_prob.extend(prob)
			y.extend(np.argmax(prob, axis=1))
		result = {}
		result['accuracy'] = accuracy_score(y_, y, sample_weight=sw)
		result['macro_auc'] = cal_macro_auc(y_, np.array(pred_prob), sw)
		result['macro_f1'] = f1_score(y_, y, average='macro', sample_weight=sw)
		result['micro_f1'] = f1_score(y_, y, average='micro', sample_weight=sw)
		for k, v in result.items():
			logger.info('%s: %s' % (k, v))
		return result


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): list of phenotype
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		X = self.phe_lists_to_X(phe_list)  # shape=(1, feature_num)
		return self.predict_prob(X).flatten()  # shape=(class_num)


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		phe_list = slice_list_with_keep_set(phe_list, self.hpo_map_rank)
		return super(ClassificationModel, self).process_query_phe_list(phe_list, phe_list_mode, hpo_dict)


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		score_vec = self.cal_score(phe_list)
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			result = sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1], reverse=True)
		else:
			result = heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1])   # [(dis_code, score), ...], shape=(dis_num, )
		if self.use_rd_mix_code:
			return [(source_code, score) for rd_code, score in result for source_code in self.rd_to_sources[rd_code]]
		return result


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


	def query_score_mat(self, phe_lists, chunk_size=None, cpu_use=None):
		"""
		Returns:
			np.ndarray: shape=(sample_num, dis_num)
		"""
		phe_lists = [self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict) for phe_list in phe_lists]
		X = self.phe_lists_to_X(phe_lists)
		sample_size = X.shape[0]
		if chunk_size is None:
			chunk_size = sample_size
		score_matrix = []  # [sample_num, class_num]
		intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
		for i in range(len(intervals) - 1):
			prob = self.predict_prob(X[intervals[i]: intervals[i + 1]])
			score_matrix.append(prob)
		score_matrix = np.vstack(score_matrix)
		return score_matrix


	def score_mat_to_results(self, phe_lists, score_mat, topk):
		return [
			self.score_vec_to_result(score_mat[i], topk) if len(phe_lists[i]) != 0 else self.query_empty(topk)
			for i in range(score_mat.shape[0])
		]


	def query_many(self, phe_lists, topk=10, chunk_size=None, cpu_use=None):
		"""
		Args:
			phe_lists (list): [[hpo1, hpo2, ...], ...]
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list: [result1, result2, ...], result=[(dis1, score1), ...], scores decreasing
		"""
		score_mat = self.query_score_mat(phe_lists, chunk_size)
		return self.score_mat_to_results(phe_lists, score_mat, topk)


class SklearnModel(ClassificationModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode=PHELIST_ORIGIN, embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,),
			dim_reductor=None, dp=1.0, up_induced_rule='max', use_rd_mix_code=False):
		super(SklearnModel, self).__init__(hpo_reader, vec_type, phe_list_mode=phe_list_mode, embed_mat=embed_mat,
			combine_modes=combine_modes, dim_reductor=dim_reductor, dp=dp, up_induced_rule=up_induced_rule,
			use_rd_mix_code=use_rd_mix_code)
		self.SAVE_FOLDER = None
		self.MODEL_SAVE_PATH = None
		self.CONFIG_JSON = None


	def init_save_path(self):
		raise NotImplementedError


	def load(self):
		self.init_save_path()
		self.clf = joblib.load(self.MODEL_SAVE_PATH)


	def save(self, clf, c):
		self.init_save_path()
		joblib.dump(clf, self.MODEL_SAVE_PATH)
		c.save(self.CONFIG_JSON)


	def delete_model(self):
		os.remove(self.MODEL_SAVE_PATH)
		os.remove(self.CONFIG_JSON)


	def change_save_folder_and_save(self, model_name=None, save_folder=None):
		old_model_path = self.MODEL_SAVE_PATH
		old_config_path = self.CONFIG_JSON
		self.name = model_name or self.name
		self.SAVE_FOLDER = save_folder or self.SAVE_FOLDER
		self.init_save_path()
		shutil.copy(old_model_path, self.MODEL_SAVE_PATH)
		shutil.copy(old_config_path, self.CONFIG_JSON)


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		return self.clf.predict_proba(X)


class TensorflowModel(ClassificationModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode=PHELIST_ORIGIN, embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,), use_rd_mix_code=False):
		super(TensorflowModel, self).__init__(
			hpo_reader, vec_type, phe_list_mode=phe_list_mode, embed_mat=embed_mat, combine_modes=combine_modes, use_rd_mix_code=use_rd_mix_code)
		self.sess = None

	def __del__(self):
		if self.sess is not None:
			self.sess.close()


if __name__ == '__main__':
	pass






