import heapq
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.utils.fixes import logsumexp
import joblib
import os
import numpy as np
from scipy.sparse import csr_matrix, vstack, load_npz, save_npz

from core.predict.config import Config
from core.predict.model import SklearnModel, ClassificationModel, Model
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PHELIST_REDUCE, PREDICT_MODE, VEC_TYPE_LOG_PROB, TRAIN_MODE
from core.utils.constant import VEC_TYPE_TF, PHELIST_ANCESTOR_DUP, VEC_TYPE_0_1, ROOT_HPO_CODE
from core.utils.utils import cal_max_child_prob_array, scale_by_min_max, get_all_ancestors_for_many, slice_list_with_keep_set, item_list_to_rank_list
from core.utils.utils import get_csr_matrix_from_dict
from core.helper.data.data_helper import DataHelper


class MNBConfig(Config):
	def __init__(self, d=None):
		super(MNBConfig, self).__init__()
		self.alpha = 1.0
		self.class_prior = None
		if d is not None:
			self.assign(d)


class MNBModel(SklearnModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_TF, phe_list_mode=PHELIST_ANCESTOR_DUP,
			model_name=None, save_folder=None, mode=PREDICT_MODE, init_para=True):
		if mode == PREDICT_MODE:
			super(MNBModel, self).__init__(hpo_reader, vec_type, phe_list_mode, None, None)
		else:
			super(MNBModel, self).__init__(hpo_reader, VEC_TYPE_TF, PHELIST_ANCESTOR_DUP, None, None)
		self.name = 'MNBModel' if model_name is None else model_name
		self.SAVE_FOLDER = save_folder
		self.clf = None
		if init_para and mode == PREDICT_MODE:
			self.load()


	def init_save_path(self):
		self.SAVE_FOLDER = self.SAVE_FOLDER or os.path.join(MODEL_PATH, self.hpo_reader.name, 'MNBModel')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_SAVE_PATH = os.path.join(self.SAVE_FOLDER, self.name + '.joblib')
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, self.name + '.json')


	def train(self, mnb_config, save_model=True):
		print(mnb_config)
		raw_X, y_ = DataHelper(self.hpo_reader).get_train_raw_Xy(PHELIST_ANCESTOR_DUP)
		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, None, mnb_config, save_model)


	def train_X(self, X, y_, sw, mnb_config, save_model=True):
		self.clf = MultinomialNB(
			alpha=mnb_config.alpha, class_prior=mnb_config.class_prior
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			self.save(self.clf, mnb_config)


# ==============================================================
class HPOProbMNBModel(Model):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, p1=0.9, p2=None,
			child_to_parent_prob='max', model_name=None, mode=TRAIN_MODE, init_para=True):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
			p1: default prob of annotation
			p2 (float or None): background prob of each HPO
			child_to_parent_prob (str): 'sum' | 'max' | 'ind'
		"""
		super(HPOProbMNBModel, self).__init__()
		self.hpo_reader = hpo_reader
		self.name = model_name or 'HPOProbMNBModel'
		self.init_save_path()
		self.phe_list_mode = phe_list_mode
		self.p1 = p1
		self.p2 = p2
		self.child_to_parent_prob = child_to_parent_prob

		self.HPO_NUM, self.DIS_NUM = self.hpo_reader.get_hpo_num(), self.hpo_reader.get_dis_num()
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		self.dis_list = self.hpo_reader.get_dis_list()
		self.hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		self.ROOT_HPO_INT = self.hpo_map_rank[ROOT_HPO_CODE]

		self.anno_hpo_set = set(self.hpo_reader.get_anno_hpo_list())

		self.dis_hpo_ances_mat = None  # (dis_num, hpo_num)
		self.bg_log_prob_ary = None  # (hpo_num,)
		self.dis_hpo_log_prob_mat = None   # (dis_num, hpo_num)

		if init_para:
			if mode == PREDICT_MODE:
				self.load()
			else:
				self.train()


	def train(self):
		self.dis_hpo_ances_mat = self.get_dis_hpo_ances_mat()
		self.bg_log_prob_ary = self.get_background_log_prob_ary()
		self.dis_hpo_log_prob_mat = DataHelper(self.hpo_reader).get_train_prob_X(
			dp=self.p1, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_LOG_PROB,
			up_induce_rule=self.child_to_parent_prob, sparse=True, dtype=np.float64)


	def cal_score(self, phe_list):
		phe_int_list = item_list_to_rank_list(phe_list, self.hpo_map_rank)
		q_hpo_mat = get_csr_matrix_from_dict({0: phe_int_list}, shape=(1, self.HPO_NUM), dtype=np.bool, t=True)
		dis_have_hpo_mat = self.dis_hpo_ances_mat.multiply(q_hpo_mat)
		dis_not_have_hpo_mat = vstack([q_hpo_mat] * self.DIS_NUM) - dis_have_hpo_mat
		prob_ary1 = dis_have_hpo_mat.multiply(self.dis_hpo_log_prob_mat).sum(axis=1).getA1()   # (dis_num,)
		prob_ary2 = dis_not_have_hpo_mat.multiply(self.bg_log_prob_ary).sum(axis=1).getA1()     # (dis_num,)
		return prob_ary1 + prob_ary2


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		score_vec = self.cal_score(phe_list)  # shape=[dis_num]
		assert np.sum(np.isnan(score_vec)) == 0  #
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


	def get_dis_hpo_ances_mat(self):
		return get_csr_matrix_from_dict(self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR),
												shape=(self.DIS_NUM, self.HPO_NUM), dtype=np.bool, t=True)


	def get_background_log_prob_ary(self):
		"""
		Returns:
			np.ndarray: (hpo_num,)
		"""
		hpo_int_2_dis_int = self.hpo_reader.get_hpo_int_to_dis_int(PHELIST_ANCESTOR)
		if self.p2 is None:
			M = np.zeros(shape=[self.HPO_NUM, ], dtype=np.float64)
			for hpo_rank, disRankList in hpo_int_2_dis_int.items():
				M[hpo_rank] = len(disRankList)
			M = np.log(M / self.DIS_NUM)
			M[np.isneginf(M)] = np.log(1 / self.DIS_NUM)  #
			return M
		return np.ones(shape=(self.HPO_NUM,), dtype=np.float64) * np.log(self.p2)


	def process_query_phe_list(self, phe_list, phe_list_mode, hpo_dict):
		phe_list = super(HPOProbMNBModel, self).process_query_phe_list(phe_list, phe_list_mode, hpo_dict)
		phe_list = slice_list_with_keep_set(phe_list, self.anno_hpo_set)
		return phe_list


	def init_save_path(self):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'HPOProbMNBModel', self.name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DIS_HPO_ANCES_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_hpo_ances_mat.npz')
		self.BG_LOG_PROB_ARY_NPY = os.path.join(self.SAVE_FOLDER, 'bg_log_prob_ary.npy')
		self.DIS_HPO_LOG_PROB_MAT_NPZ = os.path.join(self.SAVE_FOLDER, 'dis_hpo_log_prob_mat.npz')


	def save(self):
		save_npz(self.DIS_HPO_ANCES_MAT_NPZ, self.dis_hpo_ances_mat)
		np.save(self.BG_LOG_PROB_ARY_NPY, self.bg_log_prob_ary)
		save_npz(self.DIS_HPO_LOG_PROB_MAT_NPZ, self.dis_hpo_log_prob_mat)


	def load(self):
		self.dis_hpo_ances_mat = load_npz(self.DIS_HPO_ANCES_MAT_NPZ)
		self.bg_log_prob_ary = np.load(self.BG_LOG_PROB_ARY_NPY)
		self.dis_hpo_log_prob_mat = load_npz(self.DIS_HPO_LOG_PROB_MAT_NPZ)


# =====================================================================
class TreeMNBModel(MNBModel):
	def __init__(self, hpo_reader, p=0.01, vec_type=None, phe_list_mode=PHELIST_REDUCE,
			model_name=None, save_folder=None, mode=PREDICT_MODE, init_para=True):
		if mode == PREDICT_MODE:
			super(TreeMNBModel, self).__init__(hpo_reader, vec_type, phe_list_mode, init_para=False)
		else:
			super(TreeMNBModel, self).__init__(hpo_reader, None, PHELIST_REDUCE, init_para=False)
		self.name = model_name or 'TreeMNBModel'
		self.hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		self.p = p
		self.PROB_TO_NUM = 1/p
		self.ROOT_HPO_INT = self.hpo_map_rank[ROOT_HPO_CODE]
		if init_para and mode == PREDICT_MODE:
			self.load()


	def get_LogProb(self, hpo_int, hpo_int_to_prob):
		if hpo_int not in hpo_int_to_prob:
			return 0.0
		if hpo_int_to_prob[hpo_int] is not None:
			return hpo_int_to_prob[hpo_int]
		child_probs = [self.get_LogProb(child_int, hpo_int_to_prob) for child_int in self.hpo_int_dict[hpo_int].get('CHILD', [])]
		prob = 1.0
		for childProb in child_probs:
			prob *= (1-childProb)
		prob = 1 - prob
		hpo_int_to_prob[hpo_int] = prob
		return prob


	def hpo_int_list_to_csr_data(self, hpo_int_list):
		"""
		Args:
			hpo_int_list (list): [hpo_int, ...]
		Returns:
			list: col list
			list: value list
		"""
		hpo_int_to_prob = {hpo_int: None for hpo_int in get_all_ancestors_for_many(hpo_int_list, self.hpo_int_dict)}
		for hpo_int in hpo_int_list:
			hpo_int_to_prob[hpo_int] = self.p
		self.get_LogProb(self.ROOT_HPO_INT, hpo_int_to_prob)
		hpo_int_to_tf = {hpo_int: prob*self.PROB_TO_NUM for hpo_int, prob in hpo_int_to_prob.items()}
		return list(hpo_int_to_tf.keys()), list(hpo_int_to_tf.values())


	def raw_X_to_X(self, raw_X):

		value_list, row_list, col_list = [], [], []
		for i, hpo_int_list in enumerate(raw_X):
			cList, v_list = self.hpo_int_list_to_csr_data(hpo_int_list)
			value_list.extend(v_list)
			col_list.extend(cList)
			row_list.extend([i]*len(v_list))
		return csr_matrix((value_list, (row_list, col_list)), shape=(len(raw_X), self.HPO_CODE_NUMBER), dtype=np.float64)


	def train(self, mnb_config, save_model=True):
		print(mnb_config)
		raw_X, y_ = DataHelper(self.hpo_reader).get_train_raw_Xy(PHELIST_REDUCE)
		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, None, mnb_config, save_model)


# =====================================================================
class CNBConfig(Config):
	def __init__(self, d=None):
		super(CNBConfig, self).__init__()
		self.alpha = 1.0
		self.class_prior = None
		if d is not None:
			self.assign(d)


class CNBModel(SklearnModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_TF, phe_list_mode=PHELIST_ANCESTOR_DUP,
			mode=PREDICT_MODE, model_name=None, save_folder=None, init_para=True, use_rd_mix_code=False):
		super(CNBModel, self).__init__(hpo_reader, vec_type, phe_list_mode, None, None, use_rd_mix_code=use_rd_mix_code)
		self.name = 'CNBModel' if model_name is None else model_name
		self.SAVE_FOLDER = save_folder
		self.clf = None
		if init_para and mode == PREDICT_MODE:
			self.load()


	def init_save_path(self):
		self.SAVE_FOLDER = self.SAVE_FOLDER or os.path.join(MODEL_PATH, self.hpo_reader.name, 'CNBModel')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_SAVE_PATH = os.path.join(self.SAVE_FOLDER, self.name + '.joblib')
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, self.name + '.json')


	def train(self, cnb_config, save_model=True):

		print(cnb_config)
		raw_X, y_ = DataHelper(self.hpo_reader).get_train_raw_Xy(self.phe_list_mode, use_rd_mix_code=self.use_rd_mix_code)
		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, None, cnb_config, save_model)


	def train_X(self, X, y_, sw, cnb_config, save_model=True):
		self.clf = ComplementNB(
			alpha=cnb_config.alpha, class_prior=cnb_config.class_prior
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			self.save(self.clf, cnb_config)


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		return self.clf.predict_log_proba(X)


# =====================================================================
class BNBConfig(Config):
	def __init__(self, d=None):
		super(BNBConfig, self).__init__()
		self.alpha = 1.0
		self.class_prior = None
		if d is not None:
			self.assign(d)


class BNBModel(ClassificationModel):
	def __init__(self, hpo_reader=HPOReader(), default_prob=0.5, vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR,
			mode=PREDICT_MODE, model_name=None, init_para=True):
		if mode == PREDICT_MODE:
			super(BNBModel, self).__init__(hpo_reader, vec_type, phe_list_mode, None, None)
		else:
			super(BNBModel, self).__init__(hpo_reader, VEC_TYPE_TF, PHELIST_ANCESTOR_DUP, None, None)
		self.name = 'BNBModel' if model_name is None else model_name
		self.dp = default_prob

		folder = MODEL_PATH + os.sep + self.hpo_reader.name + os.sep + 'BNBModel'; os.makedirs(folder, exist_ok=True)
		self.MODEL_SAVE_PATH = folder + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = folder + os.sep + self.name + '.json'
		self.log_theta_mat = None    # shape=(dis_num, hpo_num)
		if init_para and mode == PREDICT_MODE:
			self.load()


	def train(self, raw_X, y_, sw, bnb_config, logger, save_model=True):

		print(bnb_config)
		X = self.raw_X_to_X_func(raw_X)
		X = X

		self.train_X(X, y_, sw, bnb_config, save_model)


	def train_X(self, X, y_, sw, mnb_config, save_model=True):
		self.clf = MultinomialNB(
			alpha=mnb_config.alpha, class_prior=mnb_config.class_prior
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			joblib.dump(self.clf, self.MODEL_SAVE_PATH)
			mnb_config.save(self.CONFIG_JSON)


	def load(self):
		pass


# =====================================================================
class BNBProbConfig(Config):
	def __init__(self):
		super(BNBProbConfig, self).__init__()
		self.anno_dp = 1.0
		self.not_have_dp = 0.0
		self.min_prob = 0.1
		self.max_prob = 0.9


class BNBProbModel(SklearnModel):
	def __init__(self, hpo_reader=HPOReader(), mode=PREDICT_MODE, model_name=None, init_para=True):
		super(BNBProbModel, self).__init__(hpo_reader, VEC_TYPE_0_1, PHELIST_ANCESTOR, None, None)
		self.name = 'BNBProbModel' if model_name is None else model_name

		self.clf = None
		folder = MODEL_PATH + os.sep + self.hpo_reader.name + os.sep + 'BNBProbModel'; os.makedirs(folder, exist_ok=True)
		self.FEATURE_LOG_PROB_PATH = folder + os.sep + self.name + '.npz'
		self.CONFIG_JSON = folder + os.sep + self.name + '.json'

		self.feature_log_prob = None
		if init_para and mode == PREDICT_MODE:
			self.load()

	def cal_feature_log_prob(self, bnb_config):
		def check():
			assert np.sum(np.isnan(self.feature_log_prob)) == 0
			assert np.sum(np.isneginf(self.feature_log_prob)) == 0
		dis_int_to_hpo_int_prob = self.hpo_reader.get_dis_int_to_hpo_int_prob(default_prob=bnb_config.anno_dp)
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		self.feature_log_prob = np.zeros(shape=(self.DIS_CODE_NUMBER, self.HPO_CODE_NUMBER), dtype=np.float64)
		for i in range(self.DIS_CODE_NUMBER):
			self.feature_log_prob[i, :] = cal_max_child_prob_array(dis_int_to_hpo_int_prob[i], hpo_int_dict, bnb_config.not_have_dp, dtype=np.float64)
		self.feature_log_prob = scale_by_min_max(self.feature_log_prob, bnb_config.min_prob, bnb_config.max_prob, 0.0, 1.0)
		self.feature_log_prob = np.log(self.feature_log_prob)
		check()


	def train(self, bnb_config, save_model=True):
		print(bnb_config)
		self.cal_feature_log_prob(bnb_config)
		if save_model:
			np.savez_compressed(self.FEATURE_LOG_PROB_PATH, self.feature_log_prob)
			bnb_config.save(self.CONFIG_JSON)


	def load(self):
		self.feature_log_prob = np.load(self.FEATURE_LOG_PROB_PATH)


	def predict_log_prob(self, X):
		"""ref: https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/naive_bayes.py#L89
		"""
		neg_prob = np.log(1 - np.exp(self.feature_log_prob))
		jll = X * (self.feature_log_prob - neg_prob).T + neg_prob.sum(axis=1)
		log_prob_X = logsumexp(jll, axis=1)
		return jll - np.atleast_2d(log_prob_X).T


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		return np.exp(self.predict_log_prob(X))


if __name__ == '__main__':
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	model = HPOProbMNBModel(hpo_reader)



