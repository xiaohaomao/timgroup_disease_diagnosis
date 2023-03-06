

import os
import re
import time
import json
import pickle
import numpy as np
from scipy.sparse import save_npz, load_npz
import joblib
# from multiprocessing import Pool
from billiard.pool import Pool # To avoid AssertionError: daemonic processes are not allowed to have children
from tqdm import tqdm
from queue import Queue
from zhon import hanzi
import string
from sklearn.metrics import euclidean_distances as euclidean_distances_base

from bert_syn.utils.constant import JSON_FILE_FORMAT, PKL_FILE_FORMAT, NPY_FILE_FORMAT
from bert_syn.utils.constant import  NPZ_FILE_FORMAT, SPARSE_NPZ_FILE_FORMAT, JOBLIB_FILE_FORMAT


def equal_to(a, b, eps=1e-6):
	return np.abs(a - b) < eps


def dict_set_add(k, v, d):
	if k in d:
		d[k].add(v)
	else:
		d[k] = {v}


def dict_set_update(k, vlist, d):
	if k in d:
		d[k].update(vlist)
	else:
		d[k] = set(vlist)


def split_path(path):
	"""'a/b.json' -> ('a', 'b', '.json')
	"""
	folder, fullname = os.path.split(path)
	prefix, postfix = os.path.splitext(fullname)
	return folder, prefix, postfix


def jaccard_sim(set1, set2, sym=True):
	intersect_len = len(set1 & set2)
	if sym:
		return intersect_len / (len(set1) + len(set2) - intersect_len)
	else:
		return intersect_len / len(set2)
	# return len(set1 & set2) / len(set1 | set2)


def cal_jaccard_sim_wrap(paras):
	pairs, sym = paras
	return [jaccard_sim(set(str1), set(str2), sym) for str1, str2 in pairs]


def cal_jaccard_sim_list(pairs, sym=True, cpu_use=12, chunk_size=1000):
	def get_iterator(pairs, chunk_size):
		for i in tqdm(range(0, len(pairs), chunk_size)):
			yield pairs[i: i+chunk_size], sym
	if cpu_use == 1:
		return cal_jaccard_sim_wrap((pairs, sym))
	with Pool(cpu_use) as pool:
		it = get_iterator(pairs, chunk_size)
		sim_list = []
		for sims in pool.imap(cal_jaccard_sim_wrap, it):
			sim_list.extend(sims)
		return sim_list


def cal_quartile(ary):
	"""
	Args:
		data_list (list): list of number
	Returns:
		list: [minimum, Q1, median, Q3, maximum]
	"""
	return np.percentile(ary, [0, 25, 50, 75, 100]).tolist()


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


def timer(func):
	def wrapper(*args, **kwargs):
		print('{0} starts running...'.format(func.__name__))
		start_time = time.time()
		ret = func(*args, **kwargs)
		print('Function {0} finished. Total time cost: {1} seconds'.format(func.__name__, '{:.4f}'.format(time.time()-start_time)))
		return ret
	return wrapper


def read_file_folder(path, handle_func, recursive=True):
	"""
	Args:
		path (string): path of file or file folder
		handleFunc (function): paras = (file_path)
		recursive (bool): Whether to recursively traverse the sub folders
	"""

	if os.path.isfile(path):

		handle_func(path)
	elif recursive:
		for file_name in os.listdir(path):
			file_dir = os.path.join(path, file_name)

			read_file_folder(file_dir, handle_func, recursive)



def get_file_list(path, filter):
	"""
	Args:
		dir (string): path of file or file folder
		filter (function): paras = (file_path); i.e. filter=lambda file_path: file_path.endswith('.json')
	Returns:
		list: [file_path1, file_path2, ...]
	"""
	def handle_func(file_path):
		if filter(file_path):
			file_list.append(file_path)
	file_list = []


	read_file_folder(path, handle_func)
	return file_list


def get_first_non_blank(s):
	"""''
	Args:
		s (str)
	"""
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


def write_standard_file(item_lists, file_name, split_char='\t'):
	"""
	Args:
		item_lists (list): [line_items, ...]; line_item = [item1, item2, ...]
		file_name (str)
		split_char (str)
	"""
	os.makedirs(os.path.dirname(file_name), exist_ok=True)
	open(file_name, 'w').write('\n'.join([split_char.join([str(item) for item in items]) for items in item_lists]))


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



def check_return(attr_collector):
	def outerWrapper(func):
		def wrapper(cls, *args, **kwargs):
			coll = getattr(cls, attr_collector, None)
			if coll is not None:
				return coll
			coll = func(cls, *args, **kwargs)
			setattr(cls, attr_collector, coll)
			return coll
		return wrapper
	return outerWrapper


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


def unique_pairs(pairs):
	return unique_lists(pairs)


def unique_lists(lists, items_to_key=None):
	items_to_key = items_to_key or (lambda items: tuple(sorted(items)))
	key_set = set()
	ret_list = []
	for items in lists:
		key = items_to_key(items)
		if key in key_set:
			continue
		key_set.add(key)
		ret_list.append(items)
	return ret_list


def wide_search_base(code, hpo_dict, key, contain_self):
	"""
	Args
		code (str): hpo code
		hpo_dict (dict): {hpo_code: {key: [], ...}}
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
		for pCode in hpo_dict[ex_code].get(key, []):
			if pCode not in p_set:
				p_set.add(pCode)
				q.put(pCode)
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
	"""
	Args:
		codes (list): [hpo_code1, ...]
		hpo_dict (dict): {hpo_code: {'IS_A': [], ...}}
		contain_self (bool)
	Returns:
		set: set of codes and codes' ancestors/descendents (according to key)
	"""
	union_set = set()
	for code in codes:
		union_set.update(wide_search_base(code, hpo_dict, key, contain_self))
	return union_set


def wide_search_for_many_base_fix_step(codes, hpo_dict, key, step, contain_self):
	union_set = set()
	for code in codes:
		union_set.update(wide_search_base_fix_step(code, hpo_dict, key, step, contain_self))
	return union_set


def get_all_ancestors(code, hpo_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_base(code, hpo_dict, 'IS_A', contain_self)
	return wide_search_base_fix_step(code, hpo_dict, 'IS_A', step, contain_self)


def get_all_ancestors_for_many(codes, hpo_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_for_many_base(codes, hpo_dict, 'IS_A', contain_self)
	return wide_search_for_many_base_fix_step(codes, hpo_dict, 'IS_A', step, contain_self)


def get_all_descendents(code, hpo_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_base(code, hpo_dict, 'CHILD', contain_self)
	return wide_search_base_fix_step(code, hpo_dict, 'CHILD', step, contain_self)


def get_all_descendents_for_many(codes, hpo_dict, contain_self=True, step=None):
	if step is None:
		return wide_search_for_many_base(codes, hpo_dict, 'CHILD', contain_self)
	return wide_search_for_many_base_fix_step(codes, hpo_dict, 'CHILD', step, contain_self)


PUNC_STR = hanzi.punctuation+string.punctuation+'±'
def is_punc(c):
	return PUNC_STR.find(c) > -1

def is_space(c):
	return c.isspace()  # ' ', '\n', '\t', ...

def contain_cns(s):
	return re.search('[\u4e00-\u9fff]+', s) is not None

def contain_digits(s):
	return re.search('[0-9]+', s) is not None

NEG_PATTERN = re.compile('未引出|未见|没有|否认|无|（-）|不明显|未再|未出现|不符合|不考虑|除外|未诉|未见异常|不伴')
def contain_neg(s):
	return NEG_PATTERN.search(s)


def divide_mat(X, cpu=12, min_chunk=200, max_chunk=1000):
	sample_size = X.shape[0]
	chunk_size = max(min(sample_size // cpu, max_chunk), min_chunk)
	intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
	para_list = [X[intervals[i]: intervals[i + 1]] for i in
		range(len(intervals) - 1)]
	return para_list


def euclidean_distances_wrapper(paras):
	X, Y, squared = paras
	return euclidean_distances_base(X, Y, squared=squared)


@timer
def euclidean_distances(X, Y, squared=False, cpu_use=12, min_chunk=5, max_chunk=500):
	def get_iterator(X, Y, squared, cpu_use, min_chunk, max_chunk):
		sub_X_list = divide_mat(X, cpu=cpu_use, min_chunk=min_chunk, max_chunk=max_chunk)
		for sub_X in sub_X_list:
			yield sub_X, Y, squared
	it = get_iterator(X, Y, squared, cpu_use, min_chunk, max_chunk)
	pair_dist_ary = []
	with tqdm(total=X.shape[0]) as pbar:
		if cpu_use == 1:
			for paras in it:
				ary = euclidean_distances_wrapper(paras)
				pair_dist_ary.append(ary)
				pbar.update(ary.shape[0])
		else:
			with Pool(cpu_use) as pool:
				for ary in pool.imap(euclidean_distances_wrapper, it):
					pair_dist_ary.append(ary)
					pbar.update(ary.shape[0])
	return np.vstack(pair_dist_ary)


if __name__ == '__main__':
	pass