import random
import numpy as np

from core.predict.config import Config
from core.utils.constant import DISEASE_NUM, PHELIST_ANCESTOR, VEC_TYPE_0_1
import scipy


class BatchController(object):
	def __init__(self, data_size=None, shuffle=True, seed=None):
		self.shuffle = shuffle
		if data_size is not None:
			self.set_data_size(data_size)
		self.dh = None
		self.seed = seed
		self.random = random.Random(seed)


		self.fuck = 0

	def set_data_size(self, data_size):
		print('Set data size: {}'.format(data_size))
		self.data_size = data_size
		self.rank_list = list(range(self.data_size))
		self.current_rank = self.data_size


	def reset(self):
		if self.shuffle:
			self.random.shuffle(self.rank_list)
		self.current_rank = 0


	def not_fetch_num(self):
		return self.data_size-self.current_rank


	def next_batch(self, batch_size):
		if self.not_fetch_num() < batch_size:
			self.reset()
		sample_ranks = self.rank_list[self.current_rank: self.current_rank + batch_size]
		self.current_rank += batch_size
		return sample_ranks


class BatchControllerObjList(BatchController):
	def __init__(self, obj_list, shuffle=True):
		assert len(obj_list) > 0
		super(BatchControllerObjList, self).__init__(len(obj_list), shuffle)
		self.obj_list = obj_list


	def next_batch(self, batch_size):
		"""
		Args:
			batch_size (int): batch_size
		Returns:
			list: [obj1, obj2, ...]; len = batch_size
		"""
		sample_ranks = super(BatchControllerObjList, self).next_batch(batch_size)
		return list(map(lambda i: self.obj_list[i], sample_ranks))
	

class BatchControllerMat(BatchController):
	def __init__(self, mat_list, shuffle=True, seed=None):
		"""
		Args:
			mat_list (list): [mat1, mat2, ...]; type(mat)=np.ndarray or csr_matrix; len(mat1)==len(mat2)==...
		"""
		assert len(mat_list) > 0
		super(BatchControllerMat, self).__init__(mat_list[0].shape[0], shuffle, seed)
		self.mat_list = mat_list  # [mat1, mat2, ...]
		self.check()


	def check(self):
		for mat in self.mat_list:
			assert mat.shape[0] == self.data_size


	def next_batch(self, batch_size):
		"""
		Args:
			batch_size (int): batch_size
		Returns:
			list: [mat1Rows, mat2Rows, ...]; type(mat1Rows)==type(mat1); len(matRows)==batch_size;
		"""
		sample_ranks = super(BatchControllerMat, self).next_batch(batch_size)
		return [mat[sample_ranks] for mat in self.mat_list]


class BatchControllerMixupMat(object):
	def __init__(self, alpha, bc1, bc2, seed=None):
		self.alpha = alpha
		self.bc1, self.bc2 = bc1, bc2
		self.np_random = np.random.RandomState(seed)


	def next_batch(self, batch_size):
		"""
		Returns:
			np.ndarray or csr_matrix: X; shape=(batch_size, hpo_num)
			np.ndarray or csr_matrix: y_; shape=(batch_size, dis_num)
		"""
		mats1 = self.bc1.next_batch(batch_size)
		mats2 = self.bc2.next_batch(batch_size)
		lam = self.np_random.beta(self.alpha, self.alpha)
		mats = [lam * m1 + (1. - lam) * m2 for m1, m2 in zip(mats1, mats2)]
		return mats


class MultiBatchController(object):
	def __init__(self, bc_list, kr=1.0):
		"""
		Args:
			bc_list (list): [BatchController, ...]
			kr_list (list or float): [BatchKeepRate, ...]
		"""
		self.bc_list = bc_list
		self.kr_list = kr if isinstance(kr, list) or isinstance(kr, tuple) else [kr] * len(bc_list)


	def next_batch(self, batch_size):
		"""
		Returns:
			list: [batch1, batch2, ...]; len(batch_i)==batch_size*kr_list[i];
		"""
		return [bc.next_batch(int(batch_size*self.kr_list[i])) for i, bc in enumerate(self.bc_list)]


if __name__ == '__main__':
	from tqdm import tqdm
	pass
