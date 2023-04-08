

import random
import numpy as np
from copy import copy, deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import math

from core.helper.data.batch_controller import BatchController
from core.predict.config import Config
from core.utils.constant import DISEASE_NUM, PHELIST_ANCESTOR, VEC_TYPE_0_1, PHELIST_ORIGIN, PHELIST_REDUCE
from core.utils.utils import slice_list_with_rm_set, multi_padding, get_all_ancestors_for_many, get_all_ancestors
from core.utils.utils import get_all_descendents, get_all_descendents_for_many
from core.reader.hpo_reader import HPOReader
from core.helper.data.data_helper import DataHelper


class RGBCConfig(Config):
	def __init__(self, d=None):
		super(RGBCConfig, self).__init__()
		self.multi = 20
		self.pool_update_freq = 2

		self.true = 0.5
		self.reduce = 0.2
		self.rise = 0.2
		self.lower = 0.05
		self.noise = 0.05

		self.max_reduce_prob = 0.5
		self.max_rise_prob = 0.5
		self.max_lower_prob = 0.5
		self.max_noise_prob = 0.5

		self.shuffle = True
		self.raw_xy = True
		self.phe_list_mode=PHELIST_ANCESTOR
		self.vec_type = VEC_TYPE_0_1
		self.x_sparse = True
		self.xdtype = np.int32
		self.y_one_hot = True
		self.ydtype = np.int32
		self.use_rd_mix_code = False
		self.multi_label = False

		if d is not None:
			self.assign(d)


class RandomGenBatchController(BatchController):
	def __init__(self, rgbc_config, hpo_reader=HPOReader(), cpu_use=12, seed=None):
		super(RandomGenBatchController, self).__init__(3, rgbc_config.shuffle, seed=seed)
		self.c = rgbc_config
		self.cpu_use = cpu_use
		self.go_through = self.c.pool_update_freq

		self.hpo_reader = hpo_reader
		self.hpo_int_dict = hpo_reader.get_hpo_int_dict()
		self.dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int()
		self.DIS_CODE_NUM = hpo_reader.get_dis_num()
		self.HPO_CODE_NUM = hpo_reader.get_hpo_num()

		self.ancestor_dict = {hpo: get_all_ancestors(hpo, self.hpo_int_dict) for hpo in self.hpo_int_dict}
		self.rise_dict = {hpo: slice_list_with_rm_set(self.ancestor_dict[hpo], [hpo]) for hpo in self.ancestor_dict}
		self.lower_dict = {hpo: slice_list_with_rm_set(get_all_descendents(hpo, self.hpo_int_dict), [hpo]) for hpo in self.hpo_int_dict}

		self.dh = DataHelper(hpo_reader=self.hpo_reader)
		self.source_raw_X, self.source_y_ = self.dh.get_train_raw_Xy(PHELIST_REDUCE, self.c.use_rd_mix_code, self.c.multi_label, ret_y_lists=True)
		assert len(self.source_raw_X)== self.hpo_reader.get_dis_num()

		self.raw_X, self.y_ = None, None
		self.pool_random = random.Random(seed)
		self.selected_random = np.random.RandomState(seed)

	def update_pool(self):
		def get_per_multi_nums(dis_num, expected_multi):
			floor = math.floor(expected_multi)
			return floor + (self.selected_random.rand(dis_num) < (expected_multi - floor))
		def copy_lists(ll, multi_nums):
			return [item for item, multi in zip(ll, multi_nums) for _ in range(multi)]

		dis_num = len(self.source_raw_X)
		reduce_multi_nums = get_per_multi_nums(dis_num, self.c.multi * self.c.reduce)
		rise_multi_nums = get_per_multi_nums(dis_num, self.c.multi * self.c.rise)
		lower_multi_nums = get_per_multi_nums(dis_num, self.c.multi * self.c.lower)
		noise_multi_nums = get_per_multi_nums(dis_num, self.c.multi * self.c.noise)
		true_multi_nums = get_per_multi_nums(dis_num, self.c.multi * self.c.true)
		pool_size = reduce_multi_nums.sum() + rise_multi_nums.sum() + lower_multi_nums.sum() + noise_multi_nums.sum() + true_multi_nums.sum()
		self.set_data_size(pool_size)

		raw_X, y_ = [], []

		new_raw_X, new_y_ = copy_lists(self.source_raw_X, true_multi_nums), copy_lists(self.source_y_, true_multi_nums)
		raw_X.extend(new_raw_X); y_.extend(new_y_)

		new_raw_X, new_y_ = self.gen_reduce_data_set(self.source_raw_X, self.source_y_,
			max_reduce_prob=self.c.max_reduce_prob, multi=reduce_multi_nums)
		raw_X.extend(new_raw_X); y_.extend(new_y_)

		new_raw_X, new_y_ = self.gen_replace_data_set(self.source_raw_X, self.source_y_, self.rise_dict,
			max_replace_prob=self.c.max_rise_prob, multi=rise_multi_nums)
		raw_X.extend(new_raw_X); y_.extend(new_y_)

		new_raw_X, new_y_ = self.gen_replace_data_set(self.source_raw_X, self.source_y_, self.lower_dict,
			max_replace_prob=self.c.max_lower_prob, multi=lower_multi_nums)
		raw_X.extend(new_raw_X); y_.extend(new_y_)

		new_raw_X, new_y_ = self.gen_noise_data_set(self.source_raw_X, self.source_y_, self.hpo_int_dict,
			max_noise_prob=self.c.max_noise_prob, multi=noise_multi_nums, cpu_use=self.cpu_use)
		raw_X.extend(new_raw_X); y_.extend(new_y_)

		raw_X = self.dh.hpo_int_lists_to_raw_X_with_ances_dict(raw_X, self.ancestor_dict, self.c.phe_list_mode, cpu_use=self.cpu_use)
		if self.c.raw_xy:
			self.X, self.y_ = np.array(raw_X), np.array(y_)
		else:
			self.X = self.dh.hpo_int_lists_to_X_with_ances_dict(
				raw_X, self.ancestor_dict, vec_type=self.c.vec_type, sparse=self.c.x_sparse, dtype=self.c.xdtype,
				preprocess=False, cpu_use=self.cpu_use
			)
			self.y_ = self.dh.label_lists_to_matrix(
				y_, col_num=self.hpo_reader.get_dis_num(), one_hot=self.c.y_one_hot, dtype=self.c.ydtype
			)

		self.go_through = 0


	def reduce_phe(self, phe_list, max_reduce_num):

		if max_reduce_num == 0:
			return phe_list
		keep_num = self.pool_random.randint(len(phe_list) - max_reduce_num, len(phe_list) - 1)
		return self.pool_random.sample(phe_list, keep_num)


	def replace_phe(self, phe_list, max_replace_num, nearby_dict):

		if max_replace_num == 0:
			return phe_list
		new_phe_list = copy(phe_list)
		replace_num = self.pool_random.randint(1, max_replace_num)
		for i in self.pool_random.sample(range(len(phe_list)), replace_num):
			new_phe_list[i] = self.pool_random.choice(nearby_dict[phe_list[i]]) if len(nearby_dict[phe_list[i]]) != 0 else phe_list[i]
		return new_phe_list


	def noise_phe(self, phe_list, max_noise_num, nearby_set, hpo_list):  #

		if max_noise_num == 0:
			return phe_list
		noise_num = self.pool_random.randint(1, max_noise_num)
		new_phe_list = copy(phe_list)
		while noise_num > 0:
			noise_hpo = self.pool_random.choice(hpo_list)
			if noise_hpo not in nearby_set:
				new_phe_list.append(noise_hpo)
				noise_num -= 1
		return new_phe_list


	def gen_reduce_data_set(self, raw_X, y_, max_reduce_prob=0.5, multi=100):
		"""
		Args:
			raw_X (list): [[hpo1, hpo2, ...], ...]
			y_ (list): [dis1, dis2, ..]
			multi (int or list)
		Returns:
			list: new_raw_X, [[hpo1, hpo2], ...]
			list: new_y_, [dis1, dis2, ...]
		"""
		new_y_, new_raw_X = [], []
		if isinstance(multi, int):
			multi = [multi] * len(raw_X)
		for i, multi_num in tqdm(zip(range(len(raw_X)), multi), total=len(raw_X)):
			max_reduce_num = int(len(raw_X[i]) * max_reduce_prob)
			for _ in range(multi_num):
				new_raw_X.append(self.reduce_phe(raw_X[i], max_reduce_num))
				new_y_.append(y_[i])
		return new_raw_X, new_y_


	def gen_replace_data_set(self, raw_X, y_, nearby_dict, max_replace_prob=0.5, multi=100):
		"""
		Args:
			nearby_dict (dict): {hpo: [nearby1, ...], ...}
		"""
		new_y_, new_raw_X = [], []
		if isinstance(multi, int):
			multi = [multi] * len(raw_X)
		for i, multi_num in tqdm(zip(range(len(raw_X)), multi), total=len(raw_X)):
			max_replace_num = int(len(raw_X[i]) * max_replace_prob)
			for _ in range(multi_num):
				new_raw_X.append(self.replace_phe(raw_X[i], max_replace_num, nearby_dict))
				new_y_.append(y_[i])
		return new_raw_X, new_y_


	def gen_noise_data_set(self, raw_X, y_, hpo_dict, max_noise_prob=0.5, multi=100, cpu_use=12, chunk_size=None):
		if isinstance(multi, int):
			multi = [multi] * len(raw_X)
		if cpu_use == 1:
			return self.gen_noise_data_set_single(raw_X, y_, hpo_dict, max_noise_prob, multi)
		return self.gen_noise_data_set_multi(raw_X, y_, hpo_dict, max_noise_prob, multi, cpu_use, chunk_size)


	def gen_noise_data_set_single(self, raw_X, y_, hpo_dict, max_noise_prob, multi):
		"""
		Args:
			nearby_setList (list): [set([commonAnces1, commonDes1]), ...]; length=len(dis2hpo)
			hpo_list (list): [hpo1, hpo2, ...]
		"""
		hpo_list = list(hpo_dict.keys())  #
		hpo_set = set(hpo_list)
		new_y_, new_raw_X = [], []
		assert len(raw_X) == len(multi)
		for i, multi_num in zip(range(len(raw_X)), multi):
			max_noise_num = int(len(raw_X[i]) * max_noise_prob)
			nearby_set = get_all_ancestors_for_many(raw_X[i], hpo_dict)
			nearby_set.update(get_all_descendents_for_many(raw_X[i], hpo_dict))
			for _ in range(multi_num):
				new_raw_X.append(self.noise_phe(raw_X[i], max_noise_num, nearby_set, hpo_list))
				new_y_.append(y_[i])
		return new_raw_X, new_y_


	def gen_noise_data_set_multi(self, raw_X, y_, hpo_dict, max_noise_prob, multi, cpu_use, chunk_size):
		def get_iterator():
			for i in range(len(intervals) - 1):
				yield raw_X[intervals[i]: intervals[i + 1]], y_[intervals[i]: intervals[i + 1]], hpo_dict, max_noise_prob, multi[intervals[i]: intervals[i + 1]]
			pass
		with Pool(cpu_use) as pool:
			sample_size = len(raw_X)
			new_raw_X, new_y_ = [], []
			if chunk_size is None:
				chunk_size = max(min(sample_size // cpu_use, 2000), 200)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			for tmp_raw_X, tmp_y_ in tqdm(pool.imap(self.gen_noise_data_set_multi_wrap, get_iterator()), total=len(intervals) - 1, leave=False):
				new_raw_X.extend(tmp_raw_X)
				new_y_.extend(tmp_y_)
		return new_raw_X, new_y_


	def gen_noise_data_set_multi_wrap(self, para):
		return self.gen_noise_data_set_single(*para)


	def update_pool_and_reset(self):
		self.update_pool()
		self.reset()


	def next_sample_rank(self, batch_size):
		if self.not_fetch_num() < batch_size:
			self.go_through += 1
			if self.go_through >= self.c.pool_update_freq:
				self.update_pool()
			self.reset()
		sample_rank = self.rank_list[self.current_rank: self.current_rank+batch_size]
		self.current_rank += batch_size
		return sample_rank


	def next_batch(self, batch_size):
		"""
		Args:
			batch_size (int): batch_size
		Returns:
			np.ndarray: features, shape=[batch_size, d1, d2, ...],
			np.ndarray: labels, shape=[batch_size, ]
		"""
		sample_rank = self.next_sample_rank(batch_size)
		return self.X[sample_rank], self.y_[sample_rank]


class RandomGenPaddingBatchController(RandomGenBatchController):
	def __init__(self, rgbc_config, padwith=0, hpo_reader=HPOReader(), cpu_use=12):
		super(RandomGenPaddingBatchController, self).__init__(rgbc_config, hpo_reader=hpo_reader, cpu_use=cpu_use)
		self.padwith = padwith


	def update_pool(self):
		super(RandomGenPaddingBatchController, self).update_pool()
		self.X, self.seq_len = multi_padding(self.X, self.padwith, cpu_use=self.cpu_use, chunk_size=5000)


	def next_batch(self, batch_size):
		"""
		Args:
			batch_size (int): batch_size
		Returns:
			np.ndarray: idSeqs, shape=[batch_size, max_len]
			np.ndarray: seq_len, shape=[batch_size, ]
			np.ndarray: labels, shape=[batch_size, ]
		"""
		sample_rank = self.next_sample_rank(batch_size)
		return self.X[sample_rank], self.seq_len[sample_rank], self.y_[sample_rank]

