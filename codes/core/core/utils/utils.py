import random
from copy import deepcopy
import pandas as pd
import heapq
from queue import Queue
import os
import json, pickle
import logging, logging.config
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import scipy.sparse as sp
import scipy
from sklearn.metrics import roc_curve, auc
import joblib
import time
from multiprocessing import Pool
import collections
from zhon import hanzi
import string
import re
import itertools
from collections import Counter, Iterable

from core.utils.constant import JSON_FILE_FORMAT, PKL_FILE_FORMAT, NPY_FILE_FORMAT, NPZ_FILE_FORMAT, JOBLIB_FILE_FORMAT
from core.utils.constant import VEC_COMBINE_MEAN, VEC_COMBINE_SUM, VEC_COMBINE_MAX, SPARSE_NPZ_FILE_FORMAT, SEED
from core.draw.simpledraw import simple_dist_plot
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import dabest

def is_zero(x, eps=1e-6):
	return x < eps and x > -eps


def equal(x, y, eps=1e-6):
	return is_zero(x-y, eps)


def random_string(length=32):
	return ''.join(random.sample(string.ascii_letters + string.digits, length))


def random_vec(vLen, sum_value=1.0):
	v = np.random.rand(vLen)
	return v / v.sum() * sum_value


def unique_list(ll):
	return list(np.unique(ll))


def split_path(path):
	"""'a/b.json' -> ('a', 'b', '.json')
	"""
	folder, fullname = os.path.split(path)
	prefix, postfix = os.path.splitext(fullname)
	return folder, prefix, postfix
	

def flatten(l):
	for el in l:
		if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
			yield from flatten(el)
		else:
			yield el


def dict_list_add(k, v, d):
	if k in d:
		d[k].append(v)
	else:
		d[k] = [v]


def dict_list_extend(k, v_list, d):
	if k in d:
		d[k].extend(v_list)
	else:
		d[k] = list(v_list)


def dict_set_add(k, v, d):
	if k in d:
		d[k].add(v)
	else:
		d[k] = {v}


def dict_set_update(k, values, d):
	if k in d:
		d[k].update(values)
	else:
		d[k] = set(values)


def dict_list_combine(*args):
	ret_dict = {}
	for d in args:
		for k, v_list in d.items():
			dict_list_extend(k, v_list, ret_dict)
	return ret_dict


def dict_put_if(k, v, d, ifFunc):
	if ifFunc(k, v, d):
		d[k] = v


def reverse_dict(d):
	new_dict = {}
	for k, v in d.items():
		dict_list_add(v, k, new_dict)
	return new_dict


def reverse_dict_list(d):
	new_dict = {}
	for k in d:
		for v in d[k]:
			dict_list_add(v, k, new_dict)
	return new_dict


def set_if_not_empty(d, k, v):
	if v:
		d[k] = v


def del_if_empty(d):
	for k in list(d.keys()):
		if not d[k]:
			del d[k]
	return d


def list_find(ll, fit_func):
	for i in range(len(ll)):
		if fit_func(ll[i]):
			return i
	return -1


def slice_dict_with_keep_set(old_dict, keep_key_set=None):
	return old_dict if keep_key_set == None else {key:old_dict[key] for key in keep_key_set if key in old_dict}


def slice_dict_with_keep_func(old_dict, keep_key_func):
	return {key:old_dict[key] for key in old_dict if keep_key_func(key)}


def slice_list_with_rm_set(oldList, remove_set=None):
	return oldList if remove_set == None else [item for item in oldList if item not in remove_set]


def slice_list_with_keep_func(ll, keep_func):
	return [item for item in ll if keep_func(item)]


def slice_list_with_keep_set(ll, keep_set):
	return [v for v in ll if v in keep_set]


def strip_dict_key(d):
	for k in d:
		d[k] = d[k].strip()
	return d


def fill_dict_from_dict(d1, d2):
	for k2 in d2:
		d1[k2] = d2[k2]
	return d1


def list_add_tail(l, target, num):
	if num == 0:
		return l
	l.extend([target]*num)
	return l


def del_obj_list_dup(obj_list, objToId):
	ret_list = []
	id_set = set()
	for obj in obj_list:
		id = objToId(obj)
		if id in id_set:
			continue
		ret_list.append(obj)
		id_set.add(id)
	return ret_list


def count_obj_list_dup(obj_list, objToId):
	return Counter([objToId(obj) for obj in obj_list])


def jacard(s1, s2):
	if len(s1) == 0 and len(s2) == 0:
		return 0.0
	return len(s1 & s2) / len(s1 | s2)


def gen_hpo_item_text(cns_name, eng_name, lang='zh'):
	if lang == 'en':
		cns_name = ''
	if cns_name and eng_name:
		return cns_name + ' (' + eng_name + ')'
	return (cns_name or '') + (eng_name or '')


def get_reverse_key(key):
	return 'IS_A' if key == 'CHILD' else 'CHILD'


def heap_to_list(heap):
	return [heapq.heappop(heap) for i in range(len(heap))]


def csr_like_mat(s, v, dtype=None):
	if dtype is None:
		dtype = s.dtype
	return csr_matrix(([v]*s.count_nonzero(), s.nonzero()), shape=s.shape, dtype=dtype)


def get_shortest_path_to_root(code, root, info_dict):
	"""
	Returns:
		list: [root_code, mid_code1, mid_code2, ..., code]
	"""
	ret_dict = wide_search_from_to_base_with_all_path(code, root, 'IS_A', info_dict)
	path_list = []
	p_code = root
	while p_code != code:
		path_list.append(p_code)
		p_code = ret_dict[p_code][0][1]
	path_list.append(code)
	return path_list


def get_all_path_to_root(code, root, info_dict):
	"""
	Returns:
		dict: {code: [], hpo_code1: [from_code, ...], ...};
	"""
	path_dict = wide_search_from_to_base_with_all_path(code, root, 'IS_A', info_dict)
	for hpo_code in path_dict:
		heap = path_dict[hpo_code]
		path_dict[hpo_code] = [heapq.heappop(heap)[1] for i in range(len(heap)) ]
	return path_dict


def get_all_path_str(to_code, path_dict, prefix='', sep='\t'):
	"""
	Args:
		path_dict (dict): {hpo_code: [from_code, ...]}
	Returns:
		str: string of all path come back from to_code
	"""
	ret_str = '{}{}\n'.format(prefix, to_code)
	for from_code in path_dict.get(to_code, []):
		ret_str += get_all_path_str(from_code, path_dict, prefix+sep, sep)
	return ret_str


def wide_search_from_to_base_with_all_path(from_code, to_code, key, info_dict):
	"""
	Returns:
		dict: {code: heapq([(distant, from_code), ...]) }; distant=shortest distant between hpo_code and code if comes from from_code;
	"""
	ret_dict = wide_search_base_with_all_path(from_code, key, info_dict)
	if to_code not in ret_dict:
		return {}
	r_key = get_reverse_key(key)
	path_code_set = {to_code}
	q = Queue()
	q.put(to_code)
	while not q.empty():
		ex_code = q.get()
		for p_code in info_dict[ex_code].get(r_key, []):
			if p_code in ret_dict and p_code not in path_code_set:
				q.put(p_code)
				path_code_set.add(p_code)
	return slice_dict_with_keep_set(ret_dict, path_code_set)


def wide_search_base_with_all_path(code, key, info_dict):
	"""
	Args:
		code (str)
		key (str): 'IS_A' or 'CHILD'
		info_dict (dict): {code: {'IS_A': [], 'CHILD': []}, ...}
	Returns:
		dict: {code: heapq([(distant, from_code), ...]) }; distant=shortest distant between hpo_code and code if comes from from_code;
	"""
	ret_dict = {code: []}
	once_in_queue = set()
	q = Queue()
	q.put((code, 0))
	while not q.empty():
		ex_code, dis = q.get()
		p_dis = dis + 1
		for p_code in info_dict[ex_code].get(key, []):
			if p_code not in ret_dict:
				ret_dict[p_code] = []
			heapq.heappush(ret_dict[p_code], (p_dis, ex_code))
			if p_code in once_in_queue:
				continue
			q.put((p_code, p_dis))
			once_in_queue.add(p_code)
	return ret_dict


def wide_search_base_with_shortest_dist(code, info_dict, key, contain_self):

	p_dict = {}  # codes once in queue
	q = Queue()
	q.put((code, 0))
	p_dict[code] = 0
	while not q.empty():
		ex_code, dis = q.get()
		dis += 1
		for p_code in info_dict[ex_code].get(key, []):
			if p_code not in p_dict:
				p_dict[p_code] = dis
				q.put((p_code, dis))
	if not contain_self:
		del p_dict[code]
	return p_dict


def wide_search_base(code, info_dict, key, contain_self):
	"""
	Args
		code (str): hpo code
		info_dict (dict): {code: {key: [], ...}}
		contain_self (bool)
	Returns:
		set: set of codes and codes' ancestors/descendents (according to key)
	"""
	p_set = set()    # codes once in queue
	q = Queue()     # breadth-first search queue
	q.put(code)
	p_set.add(code)
	while not q.empty():
		ex_code = q.get()
		for p_code in info_dict[ex_code].get(key, []):
			if p_code not in p_set:
				p_set.add(p_code)
				q.put(p_code)
	if not contain_self:
		p_set.remove(code)
	return p_set


def wide_search_base_fix_step(code, hpo_dict, key, step, contain_self):
	all_codes = [code]
	b, e = 0, 1
	for i in range(step):
		for j in range(b, e):
			all_codes.extend(hpo_dict[all_codes[j]].get(key, []))
		b = e
		e = len(all_codes)
	p_set = set(all_codes)
	if not contain_self:
		p_set.remove(code)
	return p_set


def wide_search_for_many_base(codes, hpo_dict, key, contain_self):

	union_set = set()
	for code in codes:
		union_set.update(wide_search_base(code, hpo_dict, key, contain_self))
	return union_set


def wide_search_for_many_base_fix_step(codes, hpo_dict, key, step, contain_self):
	union_set = set()
	for code in codes:
		union_set.update(wide_search_base_fix_step(code, hpo_dict, key, step, contain_self))
	return union_set


def read_file_folder(dir, filter_func, handle_func):

	if os.path.isfile(dir):
		if filter_func(dir):
			handle_func(dir)
	else:
		for file_name in os.listdir(dir):
			file_dir = os.path.join(dir,file_name)
			read_file_folder(file_dir, filter_func, handle_func)


def get_file_list(dir, filter):

	def handle_func(file_path):
		file_list.append(file_path)
	file_list = []
	read_file_folder(dir, filter, handle_func)
	return file_list


def get_file_list_from_dirs(dir_list, filter):
	file_list = []
	for dir in dir_list:
		file_list.extend(get_file_list(dir, filter))
	return file_list


def get_all_ancestors_with_dist(code, info_dict, contain_self=True):
	return wide_search_base_with_shortest_dist(code, info_dict, 'IS_A', contain_self)


def get_all_descendents_with_dist(code, info_dict, contain_self=True):
	return wide_search_base_with_shortest_dist(code, info_dict, 'CHILD', contain_self)


def get_all_ancestors(code, info_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_base(code, info_dict, 'IS_A', contain_self)
	return wide_search_base_fix_step(code, info_dict, 'IS_A', step, contain_self)


def get_all_ancestors_for_many(codes, info_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_for_many_base(codes, info_dict, 'IS_A', contain_self)
	return wide_search_for_many_base_fix_step(codes, info_dict, 'IS_A', step, contain_self)


def get_all_ancestors_for_many_with_ances_dict(codes, ancestor_dict):
	return np.unique([ancestor for code in codes for ancestor in ancestor_dict[code]]).tolist()


def get_all_descendents(code, info_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_base(code, info_dict, 'CHILD', contain_self)
	return wide_search_base_fix_step(code, info_dict, 'CHILD', step, contain_self)


def get_all_descendents_for_many(codes, info_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_for_many_base(codes, info_dict, 'CHILD', contain_self)
	return wide_search_for_many_base_fix_step(codes, info_dict, 'CHILD', step, contain_self)


def get_all_dup_ancestors_for_many(codes, info_dict, contain_self=True):
	"""delete redundancy and count duplicate ancestors
	Returns:
		list
	"""
	ances_dict = {}
	union_set = set()
	for code in codes:
		single_ances = get_all_ancestors(code, info_dict, contain_self=False)
		ances_dict[code] = single_ances
		union_set.update(single_ances)
	valid_codes = [code for code in codes if code not in union_set]
	ret_list = [ances for code in valid_codes for ances in ances_dict[code]]
	return ret_list + valid_codes if contain_self else ret_list


def get_all_dup_ancestors_for_many_with_ances_dict(codes, ances_dict, contain_self=True):
	tmp_ances_dict = {}
	union_set = set()
	for code in codes:
		single_ances = deepcopy(ances_dict[code]); single_ances.remove(code)
		tmp_ances_dict[code] = single_ances
		union_set.update(single_ances)
	valid_codes = [code for code in codes if code not in union_set]
	ret_list = [ances for code in valid_codes for ances in tmp_ances_dict[code]]
	return ret_list + valid_codes if contain_self else ret_list


def cal_shortest_dist(hpo1, hpo2, hpo_dict):

	if hpo1 == hpo2:
		return 0
	hpo2ances_dict = get_all_ancestors_with_dist(hpo2, hpo_dict)
	if hpo1 in hpo2ances_dict:
		return hpo2ances_dict[hpo1]
	p_dict = {}
	q = Queue() # codes once in queue
	q.put((hpo1, 0))
	p_dict[hpo1] = 0
	while not q.empty():
		ex_code, dis = q.get()
		dis += 1
		for p_code in hpo_dict[ex_code].get('IS_A', []):
			if p_code in hpo2ances_dict:
				return dis + hpo2ances_dict[p_code]
			if p_code not in p_dict:
				p_dict[p_code] = dis
				q.put((p_code, dis))
	assert False


def cal_mean_shortest_turn_dist(hpo1, hpo2, hpo_dict):

	hpo1_ances_dict = get_all_ancestors_with_dist(hpo1, hpo_dict)   # {hpo: distance}
	hpo2ances_dict = get_all_ancestors_with_dist(hpo2, hpo_dict)   # {hpo: distance}
	common_ances_set = set(hpo1_ances_dict.keys()) & set(hpo2ances_dict.keys())
	tp_set = set()   # turning point
	for hpo in common_ances_set:
		is_tp = True
		for child_hpo in hpo_dict[hpo].get('CHILD', []):
			if child_hpo in common_ances_set:
				is_tp = False
				break
		if is_tp:
			tp_set.add(hpo)
	return np.mean([hpo1_ances_dict[hpo]+hpo2ances_dict[hpo] for hpo in tp_set])


def get_logger(name, log_path=None, level=logging.INFO, mode='a'):

	formatter = logging.Formatter(fmt="%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
	logger = logging.getLogger(name)
	if len(logger.handlers) != 0:
		return logger
	logger.setLevel(level)
	if log_path is not None:
		fh = logging.FileHandler(log_path, mode=mode)
		fh.setFormatter(formatter)
		logger.addHandler(fh)
	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	return logger


def delete_logger(logger):
	while logger.handlers:
		logger.handlers.pop()


def item_list_to_rank_list(item_list, item_map_dict, auto_drop=False):
	if auto_drop:
		return [item_map_dict[item] for item in item_list if item in item_map_dict]
	return [item_map_dict[item] for item in item_list]


def dict_change_key_value(item_dict, key_map_rank, value_map_rank):
	"""
	Args:
		item_dict: {k: [v1, v2, ...]}
	"""
	return {key_map_rank[k]: item_list_to_rank_list(v_list, value_map_rank)  for k, v_list in item_dict.items()}


def data_to_01_matrix(data, col_num, dtype=np.int64):

	row_list, col_list = [], []
	for i in range(len(data)):
		row_item = np.unique(data[i])
		row_list.extend([i]*len(row_item))
		col_list.extend(row_item)
	return csr_matrix(([1] * len(col_list), (row_list, col_list)), shape=(len(data), col_num), dtype=dtype)


def data_to_tf_matrix(data, col_num, dtype=np.int64):

	sample_num = len(data)
	row_list = [i for i in range(sample_num) for _ in range(len(data[i]))]
	col_list = [x for sample in data for x in sample]
	return csr_matrix(([1]*len(col_list), (row_list, col_list)), shape=(len(data), col_num), dtype=dtype)


def data_to_cooccur_matrix(data, col_num, dtype=np.int64):

	rowcol = []
	for int_list in data:
		rowcol.extend(list(itertools.product(int_list, repeat=2)))
	row_list, col_list = zip(*rowcol)
	return csr_matrix(([1]*len(col_list), (row_list, col_list)), shape=(col_num, col_num), dtype=dtype)


def data_to_01_dense_matrix(data, col_num, dtype=np.float32):

	sample_num = len(data)
	row_list = [i for i in range(sample_num) for _ in range(len(data[i]))]
	col_list = [x for sample in data for x in sample]
	ret = np.zeros((sample_num, col_num), dtype=dtype)
	ret[row_list, col_list] = 1.0
	return ret


def data_to_tf_dense_matrix(data, col_num, dtype=np.float32):

	sample_num = len(data)
	mat = np.zeros((sample_num, col_num), dtype=dtype)
	for i in range(sample_num):
		counter = Counter(data[i])
		mat[i, list(counter.keys())] = list(counter.values())
	return mat


def gen_adj_list(hpo_dict, hpo_map_rank, file_path):
	"""https://github.com/phanein/deepwalk
	Args:
		hpo_dict (dict): {hpo_code: {'IS_A': [], 'CHILD': []}, ...}
		hpo_map_rank (dict): {hpo_code: rank}
		file_path (str)
	"""
	hpo_map_str_rank = {hpo: str(rank) for hpo, rank in hpo_map_rank.items()}
	line_list = []
	for hpo, info_dict in hpo_dict.items():
		adj_list = [hpo] + info_dict.get('IS_A', []) + info_dict.get('CHILD', [])
		line_list.append(' '.join(item_list_to_rank_list(adj_list, hpo_map_str_rank)) + '\n')
	open(file_path, 'w').writelines(line_list)


def get_edge_list(hpo_dict, hpo_map_rank, file_path):
	"""https://github.com/phanein/deepwalk
	Args:
		hpo_dict (dict): {hpo_code: {'IS_A': [], 'CHILD': []}, ...}
		hpo_map_rank (dict): {hpo_code: rank}
		file_path (str)
	"""
	hpo_map_str_rank = {hpo: str(rank) for hpo, rank in hpo_map_rank.items()}
	line_list = []
	for hpo, info_dict in hpo_dict.items():
		adj_list = [hpo] + info_dict.get('IS_A', []) + info_dict.get('CHILD', [])
		adj_list = item_list_to_rank_list(adj_list, hpo_map_str_rank)
		line_list.extend(['%s %s\n'%(adj_list[0], adj_list[i]) for i in range(1, len(adj_list))])
	open(file_path, 'w').writelines(line_list)


def mat_l2_norm(m):

	return m / np.sqrt(np.sum(np.power(m, 2), axis=1)).reshape(m.shape[0], 1)


def delete_redundacy(hpo_list, hpo_dict):

	common_ances_set = get_all_ancestors_for_many(hpo_list, hpo_dict, contain_self=False)
	return [hpo for hpo in hpo_list if hpo not in common_ances_set]


def delete_redundacy_multi_wrapper(paras):
	return delete_redundacy(paras[0], paras[1])


def delete_redundacy_for_many(hpo_lists, hpo_dict):
	paras = [(hpo_list, hpo_dict) for hpo_list in hpo_lists]
	ret = []
	with Pool() as pool:
		for reduct_hpo_list in pool.imap(delete_redundacy_multi_wrapper, paras):
			ret.append(reduct_hpo_list)
	return ret


def delete_redundacy_with_ances_dict(hpo_list, ances_dict):
	"""
	Args:
		ances_dict (dict): {hpo: ancestor_set}
	"""
	common_ances_set = set()
	for hpo in hpo_list:
		common_ances_set.update(ances_dict[hpo])
	return [hpo for hpo in hpo_list if hpo not in common_ances_set]


def change_to_redundacy_json(old_json, new_json, hpo_dict):
	"""
	Args:
		old_json (str): file path; [[hpo_list, disease or dis_list], ...]
		new_json (str): new file path
		hpo_dict (dict): {hpo_code: {'IS_A': [], 'CHILD': []}, ...}
	"""
	patients = json.load(open(old_json))
	patients = [[delete_redundacy(hpo_list, hpo_dict), dis_item] for hpo_list, dis_item in patients]
	json.dump(patients, open(new_json, 'w'), indent=2)


def delete_redundacy_for_mat(m, hpo_ances_mat):
	"""
	Args:
		m (csr_matrix): (sample_num, hpo_num)
		hpo_parent_mat (csr_matrix): (hpo_num, hpo_num)
	Returns:
		csr_matrix: (sample_num, hpo_num)
	"""
	ret = []
	for i in range(m.shape[0]):
		ances_mat = hpo_ances_mat[m[i].nonzero()[1]].sum(axis=0, dtype=np.bool)
		ret.append(m[i] - m[i].multiply(ances_mat))
	return sp.vstack(ret)


def binary_search(A, e, lo, hi):

	while (lo < hi):
		mi = int((lo + hi) / 2)
		if e > A[mi]:
			lo = mi + 1
		else:
			hi = mi
	return hi


bise_mat = np.vectorize(np.searchsorted, signature='(n),()->()')
"""binary search
Args:
	a (np.ndarray): shape=(vNum, vSize)
	v (np.ndarray): shape=(vNum,)
Returns:
	np.ndarray: index; shape=v.shape, dtype=np.int32
"""

bise_mat_with_vmat = np.vectorize(np.searchsorted, signature='(n),(m)->(m)')
"""binary search
Args:
	a (np.ndarray): shape=(vNum, vSize)
	v (np.ndarray): shape=(vNum, qryNum)
Returns:
	np.ndarray: indices; shape=v.shape, dtype=np.int32
"""


# def bise_mat(a, v):
# 	"""
# 	Args:
# 		a (np.ndarray): shape=(vNum, vSize)
# 		v (np.ndarray): shape=(vNum,)
# 	Returns:
# 		np.ndarray: shape=v.shape, dtype=np.int32
# 	"""
# 	return np.array([np.searchsorted(a[i], v[i]) for i in range(a.shape[0])])


def transform_type(coll, trans_func):

	if type(coll) == list:
		return [transform_type(item, trans_func) for item in coll]
	elif type(coll) == tuple:
		return tuple(transform_type(item, trans_func) for item in coll)
	elif type(coll) == dict:
		return {k: transform_type(v, trans_func) for k, v in coll.items()}
	else:
		return trans_func(coll)


def union_many_set(set_list):
	"""
	Args:
		set_list (list): [set1, set2, ...]
	"""
	union_set = set()
	for s in set_list:
		union_set.update(s)
	return union_set


def read_train(path):

	with open(path) as fin:
		lines = fin.readlines()
		data = [line.strip().split(' ') for line in lines]
		data = transform_type(data, lambda x: int(x))
		y_ = [sample[0] for sample in data]
		raw_X = [sample[1:] for sample in data]
		return raw_X, y_


def read_train_from_files(path_list, file_weights=None, fix=False):

	raw_X, y_, sample_nums = [], [], []
	for path in path_list:
		sub_raw_X, sub_Y_ = read_train(path)
		sample_nums.append(len(sub_raw_X))
		raw_X.extend(sub_raw_X)
		y_.extend(sub_Y_)

	total_sample_num = len(raw_X)
	if file_weights == None:
		sample_weights = np.ones(total_sample_num)
	else:
		sample_weights = np.array([file_weights[i] for i in range(len(path_list)) for _ in range(sample_nums[i])])
	if fix:
		fix_coef = np.array([sample_nums[i] for i in range(len(path_list)) for _ in range(sample_nums[i])])
		fix_coef = min(sample_nums) / fix_coef     #
		sample_weights = sample_weights * fix_coef
	return raw_X, y_, sample_weights


def cal_macro_auc(y_, y_prob, sw=None):
	"""
	Args:
		y_ (np.ndarray or list): true labels, shape=(sample_num,)
		y_prob (np.ndarray): shape=(sample_num, class_num)
		sw (np.ndarray or None): shape=(sample_num,)
	"""
	auc_list = []
	for i in range(y_prob.shape[1]):
		if sw is None:
			fpr, tpr, thresholds = roc_curve(y_, y_prob[:, i], pos_label = i, sample_weight=sw)
		else:
			fpr, tpr, thresholds = roc_curve(y_, y_prob[:, i], pos_label = i)
		auc_list.append(auc(fpr, tpr))
	return np.mean(auc_list)


def padding(X, padwith, max_len=None, xdtype=np.int32, ldtype=np.int32, cpu_use=1, chunk_size=1000):
	if cpu_use == 1:
		return single_padding(X, padwith, max_len, xdtype, ldtype)
	return multi_padding(X, padwith, max_len, xdtype, ldtype, cpu_use, chunk_size)


def single_padding(X, padwith, max_len=None, xdtype=np.int32, ldtype=np.int32):
	"""
	Args:
		(np.ndarray): np.array([[item1, item2, item3], [item4], ...]), shape=(sample_num, NotFix)
		max_len (int or None)
	Returns:
		np.ndarray: padding X, shape=(sample_num, max_len)
		np.ndarray: seq_len, shape=(sample_num,)
	"""
	seq_len = np.array([len(item_list) for item_list in X], dtype=ldtype)
	if max_len is None:
		max_len = max(seq_len)
	new_X = np.array([item_list+[padwith]*(max_len-len(item_list)) for item_list in X], dtype=xdtype)
	return new_X, seq_len


def padding_wrap(args):
	X, padwith, max_len, xdtype = args
	return np.array([item_list+[padwith]*(max_len-len(item_list)) for item_list in X], dtype=xdtype)


def multi_padding(X, padwith, max_len=None, xdtype=np.int32, ldtype=np.int32, cpu_use=12, chunk_size=1000):
	sample_size = len(X)
	seq_len = np.array([len(item_list) for item_list in X], dtype=ldtype)
	max_len = max(seq_len) if max_len is None else max_len

	intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
	para_list = [ (X[intervals[i]: intervals[i+1]], padwith, max_len, xdtype) for i in range(len(intervals) - 1)]
	sub_X_list = []
	with Pool(cpu_use) as pool:
		for subX in pool.imap(padding_wrap, para_list):
			sub_X_list.append(subX)
	return np.vstack(sub_X_list), seq_len


def pagerank(M, eps=1.0e-8, d=0.85):

	N = M.shape[1]
	v = np.random.rand(N, 1)    # shape=(N, 1); uniform distribution 0~1
	v = v / v.sum() #
	last_v = np.ones((N, 1), dtype=np.float32) * np.inf
	M_hat = d * M + ((1-d)/N)
	while np.linalg.norm(v - last_v, 1) > eps:  #
		last_v = v
		v = np.matmul(M_hat, v)
	return v.flatten()


def get_first_non_blank(s):

	for c in s:
		if not c.isspace():
			return c
	return ''


def read_standard_file(file_name, comment_char='#', split_char='\t'):

	info = []
	with open(file_name) as f:
		lines = f.read().splitlines()
		for line in lines:
			if get_first_non_blank(line) == comment_char:
				continue
			if len(line.strip()) == 0:
				continue
			info.append([item.strip() for item in line.split(split_char)])
	return info


def get_load_func(file_format):
	if file_format == JSON_FILE_FORMAT:
		return lambda path: json.load(open(path))
	if file_format == PKL_FILE_FORMAT:
		return lambda path: pickle.load(open(path, 'rb'))
	if file_format == NPY_FILE_FORMAT or file_format == NPZ_FILE_FORMAT:
		return lambda path: np.load(path)
	if file_format == SPARSE_NPZ_FILE_FORMAT:
		return lambda path: load_npz(path)
	if file_format == JOBLIB_FILE_FORMAT:
		return lambda path: joblib.load(path)
	assert False


def get_save_func(file_format):
	if file_format == JSON_FILE_FORMAT:
		return lambda obj, path: json.dump(obj, open(path, 'w'), indent=2, ensure_ascii=False)
	if file_format == PKL_FILE_FORMAT:
		return lambda obj, path: pickle.dump(obj, open(path, 'wb'))
	if file_format == NPY_FILE_FORMAT:
		return lambda obj, path: np.save(path, obj)
	if file_format == NPZ_FILE_FORMAT:
		return lambda obj, path: np.savez_compressed(path, obj)
	if file_format == SPARSE_NPZ_FILE_FORMAT:
		return lambda obj, path: save_npz(path, obj)
	if file_format == JOBLIB_FILE_FORMAT:
		return lambda obj, path: joblib.dump(obj, path)
	assert False


def load_save_for_func(file_path, file_format):
	def outer_wrapper(func):
		def wrapper(*args, **kwargs):
			if os.path.exists(file_path):
				load_func = get_load_func(file_format)
				return load_func(file_path)
			obj = func(*args, **kwargs)
			save_func = get_save_func(file_format)
			save_func(obj, file_path)
			return obj
		return wrapper
	return outer_wrapper


def check_load_save(attr_collector, attr_path, file_format):
	"""ref: http://lib.csdn.net/article/python/62942; https://blog.csdn.net/wait_for_eva/article/details/78036101
	"""
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			coll, path = getattr(cls, attr_collector), getattr(cls, attr_path)
			if coll is not None:
				return coll
			if os.path.exists(path):
				load_func = get_load_func(file_format)
				coll = load_func(path)
				setattr(cls, attr_collector, coll)
				return coll
			coll = func(cls, *args, **kwargs)
			setattr(cls, attr_collector, coll)
			save_func = get_save_func(file_format)
			os.makedirs(os.path.dirname(path), exist_ok=True)
			save_func(coll, path)
			return coll
		return wrapper
	return outer_wrapper


def load_save(attr_path, file_format):
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			path = getattr(cls, attr_path)
			if os.path.exists(path):
				load_func = get_load_func(file_format)
				coll = load_func(path)
				return coll
			coll = func(cls, *args, **kwargs)
			save_func = get_save_func(file_format)
			os.makedirs(os.path.dirname(path), exist_ok=True)
			save_func(coll, path)
			return coll
		return wrapper
	return outer_wrapper


def check_load(attr_collector, attr_path, file_format):
	"""ref: http://lib.csdn.net/article/python/62942; https://blog.csdn.net/wait_for_eva/article/details/78036101
	"""
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			coll, path = getattr(cls, attr_collector), getattr(cls, attr_path)
			if coll:
				return coll
			if os.path.exists(path):
				load_func = get_load_func(file_format)
				coll = load_func(path)
				setattr(cls, attr_collector, coll)
				return coll
			coll = func(cls, *args, **kwargs)
			setattr(cls, attr_collector, coll)
			return coll
		return wrapper
	return outer_wrapper


def check_return(attr_collector):
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			coll = getattr(cls, attr_collector)
			if coll is not None:
				return coll
			coll = func(cls, *args, **kwargs)
			setattr(cls, attr_collector, coll)
			return coll
		return wrapper
	return outer_wrapper


def timer(func):
	def wrapper(*args, **kwargs):
		print('{0} starts running...'.format(func.__name__))
		start_time = time.time()
		ret = func(*args, **kwargs)
		print('Function {0} finished. Total time cost: {1} seconds'.format(func.__name__, time.time()-start_time))
		return ret
	return wrapper


def ret_same(n):

	return n


def get_csr_matrix_from_dict(d, shape, dtype=None, t=1):
	"""
	Args:
		d (dict): {rowInt: [colInt, ...]}
		shape (tuple or list)
		t: target value
	Returns:
		csr_matrix
	"""
	row, col, data = [], [], []
	for r, cList in d.items():
		row.extend([r]*len(cList))
		col.extend(cList)
		data.extend([t]*len(cList))
	return csr_matrix((data, (row, col)), shape=shape, dtype=dtype)


def count_same_item(ordered_items):
	i, count, length = 0, 0, len(ordered_items)
	for j in range(length):
		if ordered_items[j] != ordered_items[i]:
			count += j - i if j - i > 1 else 0
			i = j
	count += length - i if length - i > 1 else 0
	return count


def count_unique_item(ordered_items):
	count, length = 0, len(ordered_items)
	if length == 0:
		return 0
	for i in range(1, length):
		if ordered_items[i] != ordered_items[i - 1]:
			count += 1
	return count + 1


def list_to_str_with_step(ll, step, list_to_str_func=lambda sub_list: '\t'.join(sub_list)):
	s = ''
	ll = [str(item) for item in ll]
	for i in range(0, len(ll), step):
		s += '{}\n'.format(list_to_str_func(ll[i: i+step]))
	return s


def vec_combine(vec_mat, row_list, mode):
	"""
	Args:
		vec_mat (np.ndarray)
		row_list (list)
		mode (str)
	Returns:
		np.ndarray
	"""
	if mode == VEC_COMBINE_MEAN:
		return np.mean(vec_mat[row_list, :], axis=0)
	if mode == VEC_COMBINE_SUM:
		return np.sum(vec_mat[row_list, :], axis=0)
	if mode == VEC_COMBINE_MAX:
		return np.max(vec_mat[row_list, :], axis=0)
	raise Exception('Unknown vector combine mode!')


def cal_max_child_prob(hpo_prob_list, hpo_dict):
	"""
	Args:
		hpo_prob_list (list): [(hpo, prob), ...]; hpo=hpo_code | hpo_int
		hpo_dict (dict): {hpo: {'IS_A': [hpo, ...], 'CHILD': [hpo, ...], ..}}
	Returns:
		dict: {hpo: prob}
	"""
	record_dict = {} # {hpo: prob}
	for hpo, prob in hpo_prob_list:
		ances_set = get_all_ancestors(hpo, hpo_dict)
		for ances in ances_set:
			dict_put_if(ances, prob, record_dict, lambda k, v, d: (k not in d) or (v > d[k]))
	return record_dict


def cal_max_child_prob_array(hpo_int_prob_list, hpo_int_dict, dp, dtype=np.float32):
	"""
	Args:
		hpo_int_prob_list (list): [(hpo_int, prob), ...]
		hpo_int_dict (dict): {hpo_rank: {'IS_A': [hpoRank1, ...], 'CHILD': [hpoRank2, ...], ..}}
		dp (float): default probability
	Returns:
		np.ndarray: shape=(hpo_num, )
	"""
	hpo_prob_dict = cal_max_child_prob(hpo_int_prob_list, hpo_int_dict)
	hpo_num = len(hpo_int_dict)
	a = np.array([dp]*hpo_num, dtype=dtype)
	ranks, values = zip(*hpo_prob_dict.items())
	a[list(ranks)] = values
	return a


def scale_by_min_max(x, new_min, new_max, old_min, old_max):
	return new_min + (new_max - new_min) * (x - old_min) / (old_max - old_min)


def get_around_adj_mat(adj_mat, order_num, dtype=np.int32, contain_self=True):

	point_num = adj_mat.shape[0]
	if order_num == 0:
		return sp.identity(point_num, dtype=np.int32)
	adj_mat += sp.identity(point_num)
	for i in range(order_num-1):
		adj_mat *= adj_mat
	adj_mat[adj_mat>1] = 1
	if not contain_self:
		adj_mat[range(point_num), range(point_num)] = 0
	return adj_mat.astype(dtype)


def get_brothers(hpo, hpo_dict, order_num):
	"""
	Returns:
		set: {hpo1, hpo2}
	"""
	ances_to_dist = get_all_ancestors_with_dist(hpo, hpo_dict)
	max_dist = max(ances_to_dist.values())
	step = min(order_num, max_dist)
	ancestors = [ancestor for ancestor, dist in ances_to_dist.items() if dist == step]

	brothers = set()
	for ancestor in ancestors:
		brothers.update(get_all_descendents(ancestor, hpo_dict, step=step))
	brothers.remove(hpo)
	return brothers


def get_brother_adj_mat(hpo_num, hpo_int_dict, order_num, dtype=np.int32, contain_self=True):
	"""
	Args:
		hpo_int_dict (dict): {hpo_int: {'CHILD': [], 'IS_A': []}}
		order_num (int)
	Returns:
		csr_matrix: every
	"""
	adj_mat = data_to_01_matrix([list(get_brothers(hpo_int, hpo_int_dict, order_num)) for hpo_int in range(hpo_num)], hpo_num, dtype)
	if contain_self:
		adj_mat += sp.identity(hpo_num, dtype=dtype)
	return adj_mat


def sparse_row_normalize(m):
	"""
	Args:
		m (csr_matrix)
	"""
	r_inv = np.power(m.sum(axis=1), -1.0).A.flatten()
	r_inv[np.isinf(r_inv)] = 0.
	return sp.diags(r_inv) * m


def sparse_element_max(m, axis):
	offset = np.max(m.data * np.sign(m.data)) + 1
	m.data += offset
	ret = m.max(axis=axis)
	m.data -= offset
	ret.data -= offset
	return ret


def sparse_element_min(m, axis):
	offset = np.max(m.data * np.sign(m.data)) + 1
	m.data -= offset
	ret = m.min(axis=axis)
	m.data += offset
	ret.data += offset
	return ret


def dense_row_normalize(m):
	"""
	Args:
		m (np.ndarray)
	"""
	r_inv = np.power(m.sum(axis=1), -1.0)    # 1-d
	r_inv[np.isinf(r_inv)] = 0.
	return np.reshape(r_inv, (r_inv.shape[0], 1)) * m


def is_jsonable(x):
	try:
		json.dumps(x)
		return True
	except:
		return False


def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
	"""
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx


def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
	"""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
	"""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return sparse_to_tuple(adj_normalized)


def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation
	copy from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
	"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return sparse_to_tuple(features)


def remove_begin_end(term):

	invalid = '([\s{cns_punc}{eng_punc}]|(NOS))+'.format(cns_punc='。，：！？；、', eng_punc='.,:!?;')
	term = re.sub('^'+invalid, '', term)
	term = re.sub(invalid+'$', '', term)
	return term


def remove_bracket(term):

	bracket_patten = '\[.*?\]|\(.*?\)|（.*?）|【.*?】|〔.*?〕'
	bracket_terms = [s[1:-1] for s in re.findall(bracket_patten, term)]
	return re.sub(bracket_patten, '', term), bracket_terms

def contain_punc(term):
	return re.search('[{cns_punc}{eng_punc}]+'.format(cns_punc=hanzi.punctuation, eng_punc=string.punctuation), term) is not None

def contain_cns(s):
	return re.search('[\u4e00-\u9fff]+', s) is not None

def all_cns(s):
	return re.match('^[\u4e00-\u9fff]+$', s) is not None

PUNC_STR = hanzi.punctuation+string.punctuation+'±'
def is_punc(c):
	return PUNC_STR.find(c) > -1

def is_space(c):
	return c.isspace()  # ' ', '\n', '\t', ...


def xlsx_to_list(path):
	"""
	Returns:
		list: [row1, row2, ...]; row={col1: value1, col2: value2}
	"""
	df = pd.read_excel(path)
	return df.to_dict(orient='records')


def xlsx_to_json(inpath, outpath):
	json.dump(xlsx_to_list(inpath), open(outpath, 'w'), indent=2, ensure_ascii=False)


def merge_pos_list(pos_list):
	"""
	Args:
		pos_list (list): [(begin1, end1), ...]; sorted by begin
	Returns:
		list: [(begin, end), ...]
	"""
	ret_pos_list = []
	last_pos = (0, 0)
	for b, e in pos_list:
		if b < last_pos[1]:
			last_pos = (last_pos[0], max(last_pos[1], e))
		else:
			ret_pos_list.append(last_pos)
			last_pos = (b, e)
	ret_pos_list.append(last_pos)
	return ret_pos_list[1:]


def edit_distance(s, t):

	if s == t:
		return 0
	if len(s)== 0:
		return len(t)
	if len(t) == 0:
		return len(s)

	v0, v1 = list(range(len(t)+1)), [0]*(len(t)+1)
	for i in range(len(s)):
		v1[0] = i+1
		for j in range(len(t)):
			if s[i] == t[j]:
				v1[j+1] = v0[j]
			else:
				v1[j+1] = min(v0[j], v0[j+1], v1[j]) + 1
		for j in range(len(v0)):
			v0[j] = v1[j]
	return v1[len(t)]


def str_list_product(*args):
	"""input=(['aa', 'bb'], ['cc'], ['dd', 'ee']); output=['aaccdd', 'aaccee', 'bbccdd', 'bbccee']
	"""
	return [''.join(str_list) for str_list in itertools.product(*args)]


def zip_sorted(*args, **kwargs):
	return zip(*sorted(zip(*args), **kwargs))


def to_rank_scores(v_list, ascending=True):
	r = pd.Series(v_list).rank(ascending=ascending)
	return (r / r.max()).tolist()


def to_zscores(v_list):
	v_array = v_list if isinstance(v_list, np.ndarray) else np.array(v_list)
	std = v_array.std() or 1e-6
	return (v_array - v_array.mean()) / std


def flatten_dict(d, prefix='', sep='-'):
	if not isinstance(d, dict):
		return {prefix: d}
	ret = {}
	for k, v in d.items():
		p = prefix + sep + k if prefix else k
		ret.update(flatten_dict(v, p, sep))
	return ret


def min_max_norm(X, feature_range=(0, 1), axis=None):
	if axis == 1:
		return min_max_norm(X.T, feature_range, 0).T
	x_max, x_min = X.max(axis=axis), X.min(axis=axis)
	x_std = (X - x_min) / (x_max - x_min)
	x_norm = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
	x_norm = np.nan_to_num(x_norm)
	return x_norm


def n_largest_indices(a, n):
	flat = a.flatten()
	indices = np.argpartition(flat, -n)[-n:]
	indices = indices[np.argsort(-flat[indices])]
	return np.unravel_index(indices, a.shape)


def n_smallest_indices(a, n):
	return n_largest_indices(-a, n)


def gen_mask_ary(n, max_len, dtype=np.bool):
	a = np.zeros(shape=(max_len,), dtype=dtype)
	a[:n] = True
	return a

"""
Args:   
	np.ndarray: lengths
	int: max_len
	dtype
Returns:
	np.ndarray: shape=(list(lengths.shape)+[max_len])
"""
sequence_mask = np.vectorize(gen_mask_ary, otypes=[np.ndarray], excluded=['dtype'])


def ary_ele_to_with_dict(v, d):
	return d[v]
modify_ary_ele_with_dict = np.vectorize(ary_ele_to_with_dict, excluded=['d'])


def combine_embed(embed_mat, id_lists, combine_mode='avg', id_weights=None):
	"""
	Args:
		embed_mat (np.ndarray): (sample_num, embed_size)
		id_lists (list): [[id1, id2, ...], ...]
		combine_mode (str): 'avg' | 'weight'
		id_weights (np.ndarray): (sample_num,)
	Returns:
		np.ndarray: (len(id_lists), embed_size)
	"""
	ret_mat = np.zeros((len(id_lists), embed_mat.shape[1]), dtype=np.float32)
	if combine_mode == 'avg':
		for i in range(len(id_lists)):
			ret_mat[i] = embed_mat[id_lists[i]].sum(axis=0) / len(id_lists[i])
	elif combine_mode == 'weight':
		for i in range(len(id_lists)):
			w = id_weights[id_lists[i]]
			w /= w.sum()
			ret_mat[i] = (embed_mat[id_lists[i]] * w.reshape((-1, 1))).sum(axis=0)
	else:
		assert False
	return ret_mat


def import_R():
	from rpy2 import robjects
	from rpy2.robjects.packages import importr
	return robjects, importr


def convert_kwargs_for_R(func):
	def f(**kwargs):
		d = {}
		for k, v in kwargs.items():
			d[k.replace('_', '.')] = v
		return func(**d)
	return f


def check_input_ary_R(a):
	if isinstance(a, np.ndarray):
		return a.tolist()
	return a


def py_wilcox(x, y=None, alternative='two.sided', mu=0, paired=False, exact=None,
		correct=True, conf_int=True, conf_level=0.95, robjects=None):
	"""ref: https://www.jianshu.com/p/c18e8d8dab88; https://rpy2.github.io/doc/latest/html/introduction.html#getting-started
	Returns:
		dict
	"""
	if robjects is None:
		from rpy2 import robjects
	x, y = check_input_ary_R(x), check_input_ary_R(y)

	x = robjects.FloatVector(x)
	wilcox_test = convert_kwargs_for_R(robjects.r['wilcox.test'])

	if y is not None:
		y = robjects.FloatVector(y)
		if exact is not None:
			pr = wilcox_test(x=x, y=y, alternative=alternative, mu=mu,
				paired=paired, exact=exact, correct=correct, conf_int=conf_int, conf_level=conf_level)
		else:
			pr = wilcox_test(x=x, y=y, alternative=alternative, mu=mu,
				paired=paired, conf_int=conf_int, conf_level=conf_level)
	else:
		if exact is not None:
			pr = wilcox_test(x=x, alternative=alternative, mu=mu,
				exact=exact, correct=correct, conf_int=conf_int, conf_level=conf_level)
		else:
			pr = wilcox_test(x=x, alternative=alternative, mu=mu,
				correct=correct, conf_int=conf_int, conf_level=conf_level)
	res_values = list(pr)
	return {
		'p_value': list(res_values[2])[0],
		'location_shift': list(res_values[3])[0],
		'alternative': list(res_values[4])[0],
		'test_name': list(res_values[5])[0],
		'conf_int': list(res_values[7]),
		'diff_in_location': list(res_values[8])[0]
	}


def cal_portion_conf_int_R(k, n, conf_level=0.95, exact=False, correct=True, alternative='two.sided', robjects=None):
	"""
	Reference:
		https://www.dxy.cn/bbs/newweb/pc/post/36625644
		http://vassarstats.net/clin1.html#note
		https://www.jiqizhixin.com/articles/2018-07-06-5
		https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_One-Sample_Sensitivity_and_Specificity.pdf
		https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
		https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
		https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/
		http://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717_ConfidenceIntervals-OneSample/PH717_ConfidenceIntervals-OneSample_print.html
	Args:
		k (int)
		n (int)
	Returns:
		(float, float)
	"""
	if robjects is None:
		from rpy2 import robjects
	if exact:
		binom_test = convert_kwargs_for_R(robjects.r['binom.test']) # Clopper and Pearson (exact)
		res = binom_test(x=k, n=n, conf_level=conf_level)
		return tuple(res[3])
	else:
		# https://stackoverflow.com/questions/41580829/what-formula-does-prop-test-use
		# https://stats.stackexchange.com/questions/183225/confidence-interval-from-rs-prop-test-differs-from-hand-calculation-and-resul

		prop_test = convert_kwargs_for_R(robjects.r['prop.test'])
		res = prop_test(x=k, n=n, conf_level=conf_level, correct=correct, alternative=alternative)
		return tuple(res[5])


def cal_median_diff_conf_int():
	pass


def cal_portion_conf_int(k, n, conf_level=0.95, method='normal'):
	from statsmodels.stats.proportion import proportion_confint
	return proportion_confint(k, n, alpha=1.-conf_level, method=method) # wilson without correction


def cal_hodges_lehmann_median_conf_int(x, conf_level=0.95, exact=False, correct=True, alternative='two.sided', robjects=None):

	x = check_input_ary_R(x)
	return tuple(py_wilcox(x, alternative=alternative, exact=exact, correct=correct,
		conf_level=conf_level, conf_int=True, robjects=robjects)['conf_int'])


def cal_boot_conf_int(x, stat_func, iter_num=10000, conf_level=0.95, stat_kwargs=None):
	"""
	References:
		https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
		https://www.jiqizhixin.com/articles/2018-07-06-5
		https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
		https://github.com/facebookincubator/bootstrapped
		https://acclab.github.io/DABEST-python-docs/bootstraps.html#
		https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
	Returns:
		(float, float)
	"""
	x = np.array(x)
	n = len(x)
	stat_kwargs = stat_kwargs or {}
	stat_results = []
	lower_p = (1. - conf_level) / 2.
	upper_p = conf_level + lower_p
	for i in range(iter_num):
		boot_x = np.random.choice(x, n, replace=True)
		stat_value = stat_func(boot_x, **stat_kwargs)
		stat_results.append(stat_value)
	return tuple(np.percentile(stat_results, [lower_p*100., upper_p*100.]).tolist())


def cal_boot_conf_int_for_multi_x(x_list, stat_func, iter_num=10000, conf_level=0.95, stat_kwargs=None):

	x_list = [np.array(x) for x in x_list]
	n_list = [len(x) for x in x_list]
	stat_kwargs = stat_kwargs or {}
	stat_results = []
	lower_p = (1. - conf_level) / 2.
	upper_p = conf_level + lower_p
	for i in range(iter_num):
		boot_x_list = [np.random.choice(x, n, replace=True) for x, n in zip(x_list, n_list)]
		stat_value = stat_func(boot_x_list, **stat_kwargs)
		stat_results.append(stat_value)
	return tuple(np.percentile(stat_results, [lower_p*100., upper_p*100.]).tolist())


def cal_boot_pvalue(x, y, statistic='mean', alternative='two.sided', iter_num=10000):
	"""H1: x != y
	References:
		https://www.youtube.com/watch?v=9STZ7MxkNVg
		https://www.youtube.com/watch?v=Zet-qmEEfCU&feature=youtu.be
		https://statslectures.com/r-scripts-datasets
	Returns:
		float
	"""
	if alternative == 'less':
		x, y = y, x

	if statistic == 'mean':
		statistics_func = abs_mean_diff if alternative == 'two.sided' else mean_diff
	elif statistic == 'median':
		statistics_func = abs_median_diff if alternative == 'two.sided' else median_diff
	else:
		assert False

	xy = np.hstack([x, y])
	stat_obs = statistics_func(x, y)
	count = 0
	for _ in range(iter_num):
		bs = np.random.choice(xy, len(xy), replace=True)
		if statistics_func(bs[:len(x)], bs[len(x):]) >= stat_obs:
			count += 1
	return count / iter_num


def mean_diff(a, b):
	return np.mean(a) - np.mean(b)
def median_diff(a, b):
	return np.median(a) - np.median(b)
def abs_mean_diff(a, b):
	return np.abs(np.mean(a) - np.mean(b))
def abs_median_diff(a, b):
	return np.abs(np.median(a) - np.median(b))


def cal_paired_boot_pvalue(x, y, metric_func, iter_num=10000):

	assert len(x) == len(y)
	n = len(x)
	x, y = np.array(x), np.array(y)
	stats_obs = metric_func(y) - metric_func(x)
	count = 0
	for _ in range(iter_num):
		sample_ranks = np.random.randint(n, size=n)
		# print('bs:', metric_func(y[sample_ranks]), metric_func(x[sample_ranks]), metric_func(y[sample_ranks]) - metric_func(x[sample_ranks]), stats_obs)
		if metric_func(y[sample_ranks]) - metric_func(x[sample_ranks]) > 2 * stats_obs:
			count += 1
	return count / iter_num


def dabest_cal_permutation_pvalue(x, y, statistic='mean', paired=False, iter_num=10000):
	"""
	References:

	Args:
		statistic (str): 'mean' | 'median'
	"""
	df = pd.DataFrame({'x': x, 'y': y, 'id': list(range(len(x)))})
	if paired:
		my_dabest_object = dabest.load(df, idx=('x', 'y'), paired=True, id_col='id', resamples=iter_num)
		pass
	else:
		my_dabest_object = dabest.load(df, idx=("x", "y"), paired=False, resamples=iter_num)
	if statistic == 'mean':
		return my_dabest_object.mean_diff.results['pvalue_permutation'].values[0]
	elif statistic == 'median':
		return my_dabest_object.median_diff.results['pvalue_permutation'].values[0]
	else:
		raise RuntimeError('Unkonwn statistics: {}'.format(statistic))


def cal_permutation_pvalue(x, y, statistic='mean', alternative='two.sided', iter_num=10000):
	"""
	References:
		https://yq.aliyun.com/articles/623828
		https://datascience103579984.wordpress.com/2019/08/17/statistical-thinking-in-python-part-2-from-datacamp/
		https://en.wikipedia.org/wiki/Resampling_(statistics)
	"""
	if alternative == 'less':
		x, y = y, x

	if statistic == 'mean':
		statistics_func = abs_mean_diff if alternative == 'two.sided' else mean_diff
	elif statistic == 'median':
		statistics_func = abs_median_diff if alternative == 'two.sided' else median_diff
	else:
		assert False

	xy = np.hstack([x, y])
	stat_obs = statistics_func(x, y)
	count = 0
	for _ in range(iter_num):
		bs = np.random.permutation(xy)
		if statistics_func(bs[:len(x)], bs[len(x):]) >= stat_obs:
			count += 1
	return count / iter_num


def cal_mcnemar_p_value(table):
	"""
	References:
		https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
		http://www.atyun.com/25532.html
		https://en.wikipedia.org/wiki/McNemar%27s_test
	Returns:
		float
	"""
	if table[0][1] + table[1][0] > 25:
		return mcnemar(table).pvalue
	return mcnemar(table, exact=True).pvalue


def cal_fisher_exact_test_p_value(table, alternative='two-sided'):
	"""
	References:
		https://www.datascienceblog.net/post/statistical_test/contingency_table_tests/
	"""
	oddsratio, pvalue = stats.fisher_exact(table, alternative)
	return pvalue


def cal_chi_exact_test_p_value(table, correction=True):
	"""
	References:
		https://blog.csdn.net/sinat_38682860/article/details/81943297
	"""
	return stats.chi2_contingency(table, correction=correction)[1]


def pvalue_correct(pvals, method='fdr_bh'):
	"""
	Args:
		pvals (list): p-value list
		method (str): 'bonferroni' | 'fdr_bh' | 'fdr_by' | 'holm' | 'hommel' | ...
	Returns:
		list: p-value list
	"""
	reject, pvals_correct, _, _ =  multipletests(pvals, alpha=0.05, method=method)
	return pvals_correct


def combine_key_to_list(d1, d2):
	d = deepcopy(d1)
	for k, vlist in d2.items():
		if k in d:
			print('Combine {}: {} + {}'.format(k, vlist, d[k]))
		dict_list_extend(k, vlist, d)
	return d


if __name__ == '__main__':
	pass
