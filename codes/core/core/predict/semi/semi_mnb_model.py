import os
import numpy as np
from tqdm import tqdm

from core.predict.config import Config
from core.predict.model import ClassificationModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PREDICT_MODE
from core.utils.constant import VEC_TYPE_TF, PHELIST_ANCESTOR_DUP, VEC_TYPE_0_1
from core.helper.data.data_helper import DataHelper


class SemiMixMNConfig(Config):
	def __init__(self, d=None):
		super(SemiMixMNConfig, self).__init__()

		self.beta = 1.0
		self.pi_init = 'same'
		self.UInit = 'TF'
		self.UAlpha = 0.01
		self.max_iter = 30
		self.tol = 100.0
		self.n_init = 1
		if d is not None:
			self.assign(d)


class SemiMixMNModel(ClassificationModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, predict_pi='same',
			mode=PREDICT_MODE, model_name=None, init_para=True):
		"""
		Args:
			predict_pi (str): 'same' | 'tune'
		"""
		super(SemiMixMNModel, self).__init__(hpo_reader, vec_type, phe_list_mode, None, None)
		self.name = 'SemiMNBModel' if model_name is None else model_name
		self.predict_pi = predict_pi

		self.U, self.pi = None, None
		self.SAVE_FOLDER = MODEL_PATH + os.sep + 'SemiMixMNModel'
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.CONFIG_JSON = self.SAVE_FOLDER + os.sep + self.name + '.json'
		self.MODEL_SAVE_NPZ = self.SAVE_FOLDER + os.sep + self.name + '.npz'
		if init_para and mode == PREDICT_MODE:
			self.load()


	def set_predict_pi(self):
		if self.predict_pi == 'same':
			self.pi = self.init_pi('same')


	def row_norm(self, M):
		return M / np.sum(M, axis=1, keepdims=True)


	def col_norm(self, M):
		return M / np.sum(M, axis=0, keepdims=True)


	def norm(self, v):
		return v / np.sum(v)


	def init_pi(self, type='same', dtype=np.float32):
		"""
		Args:
			type (str): 'same' | 'random'
		Returns:
			np.ndarray: shape=(dis_num, 1)
		"""
		K = self.DIS_CODE_NUMBER
		if type == 'same':
			return np.ones((K, 1), dtype=dtype) / K
		elif type == 'random':
			return self.norm(np.random.rand(K, 1).astype(dtype))
		assert False


	def init_U(self, type='TF', alpha=0.01, dtype=np.float32):
		"""
		Args:
			type (str): 'TF' | 'random' | 'same'
		Returns:
			np.ndrray: shape=(hpo_num, dis_num)
		"""
		K, W = self.DIS_CODE_NUMBER, self.HPO_CODE_NUMBER
		if type == 'same':
			return np.ones((W, K), dtype=dtype) / W
		elif type == 'random':
			return  self.col_norm(np.random.rand(W, K).astype(dtype))
		assert type == 'TF'
		hpo_dis_mat = DataHelper().get_train_X(PHELIST_ANCESTOR_DUP, VEC_TYPE_TF, sparse=False, dtype=dtype).T
		return self.col_norm(hpo_dis_mat+alpha)


	def init_G(self, lrows, labels, D, dtype=np.float32):
		K = self.DIS_CODE_NUMBER
		G = np.zeros((D, K), dtype=dtype)
		G[lrows, labels] = 1.0
		return G


	def cal_log_fac_table(self, n):
		ret = [0.0] * (n+1)
		for i in range(1, n+1):
			ret[i] = ret[i-1] + np.log(i)
		return ret


	def cal_log_nd_fac_T(self, T):
		nd = T.sum(axis=1).A
		max_n = nd.max()
		log_fac_table = self.cal_log_fac_table(max_n)
		for i in range(T.shape[0]):
			nd[i, 0] = log_fac_table[nd[i, 0]]
		return nd.T


	def cal_log_td_fac_T(self, T):
		max_tdw = T.max()
		log_fac_table = self.cal_log_fac_table(max_tdw)
		log_td_fac_T = np.zeros((1, T.shape[0]), dtype=np.float32)
		for d in range(T.shape[0]):
			log_td_fac_T[0, d] = sum([log_fac_table[v] for v in T[d].data])
		return log_td_fac_T


	def Q_func(self, G, T, U, pi, log_nd_fac_T, log_td_fac_T):
		Q1 = G.dot(np.log(pi)).sum()
		Q2 = (T.dot(np.log(U)) * G).sum()
		Q3 = log_nd_fac_T.dot(G).sum()
		Q4 = -log_td_fac_T.dot(G).sum()
		return Q1 + Q2 + Q3 + Q4


	def E_step(self, T, U, pi):
		temp_M = T.dot(np.log(U)) + np.log(pi.T)
		G = self.row_norm(np.exp(temp_M - np.max(temp_M, axis=1, keepdims=True)))
		return G


	def M_step(self, T, G, alpha):
		pi = self.col_norm(np.sum(G, axis=0).T)
		U = self.col_norm(T.T.dot(G) + alpha)
		return U, pi


	def EM(self, T, y_, c, log_step=5):
		D, W = T.shape  # D * W
		lrows = np.where(y_ != -1)[0]
		labels = y_[lrows].flatten()
		urows = np.where(y_ == -1)[0]

		G = self.init_G(lrows, labels, D)
		U = self.init_U(c.UInit, c.UAlpha)
		pi = self.init_pi(c.pi_init)
		log_nd_fac_T = self.cal_log_nd_fac_T(T)
		log_td_fac_T = self.cal_log_td_fac_T(T)
		beta = np.ones((D,1), dtype=np.float32); beta[urows] = c.beta

		print('Q:', self.Q_func(G, T, U, pi, log_nd_fac_T, log_td_fac_T))
		last_Q, Q = -np.inf, -np.inf
		for i in range(1, c.max_iter + 1):
			G[urows] = self.E_step(T[urows], U, pi)
			U, pi = self.M_step(T, beta * G, c.UAlpha)
			Q = self.Q_func(G, T, U, pi, log_nd_fac_T, log_td_fac_T)
			if log_step is not None and i % log_step == 0:
				print('iter {}, Q = {}, Q Diff = {}'.format(i, Q, Q - last_Q))
			if Q - last_Q < c.tol:
				print('Q - last_Q < tol({})'.format(c.tol))
				break
			last_Q = Q
		return U, pi, Q


	def train_X(self, X, y_, c, save_model=True, log_step=None):
		print('X:', X.shape, 'y_:', y_.shape)
		best_Q, bestU, best_pi = -np.inf, None, None
		for i in range(1, c.n_init+1):
			U, pi, Q = self.EM(X, y_, c, log_step)
			if Q > best_Q:
				best_Q, bestU, best_pi = Q, U, pi
			print('Init Times = {}, best_Q = {}'.format(i, best_Q))
		self.U, self.pi = bestU, best_pi
		if save_model:
			self.save()
		self.set_predict_pi()


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		return self.E_step(X, self.U, self.pi)


	def save(self):
		np.savez(self.MODEL_SAVE_NPZ, U=self.U, pi=self.pi)


	def load(self):
		data = np.load(self.MODEL_SAVE_NPZ)
		self.U, self.pi = data['U'], data['pi']
		self.set_predict_pi()


if __name__ == '__main__':
	pass


