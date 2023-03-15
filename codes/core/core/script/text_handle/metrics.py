

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_ancestors_for_many as get_all_ancestors_for_many
from core.utils.utils import get_all_ancestors as  get_all_ancestors
from core.utils.utils import get_file_list as get_file_list
from core.utils.utils import dict_list_add as dict_list_add
from core.utils.constant import DATA_PATH, RESULT_PATH


def get_hpo_dict():
	return HPOReader().get_hpo_dict()


def get_old2new_hpo():
	return HPOReader().get_old_map_new_hpo_dict()


def set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=None):
	"""
	Args:
		true_hpo_set (set): {hpo_code1, ...}
		pred_hpo_set (set): {hpo_code1, ...}
		ances_expand (bool)
		hpo_dict (dict)
	Returns:
		float
	"""
	if ances_expand:
		hpo_dict = hpo_dict or get_hpo_dict()
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(pred_hpo_set)


def set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=None):
	if ances_expand:
		hpo_dict = hpo_dict or get_hpo_dict()
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(true_hpo_set)


def set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=None):
	if ances_expand:
		hpo_dict = hpo_dict or get_hpo_dict()
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	precision = set_precision(true_hpo_set, pred_hpo_set, ances_expand=False)
	recall = set_recall(true_hpo_set, pred_hpo_set, ances_expand=False)
	return 2 * (precision * recall) / (precision + recall)


def set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=None):
	if ances_expand:
		hpo_dict = hpo_dict or get_hpo_dict()
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(true_hpo_set | pred_hpo_set)


def set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict=None, hpo2ancestors=None):
	"""
	Args:
		true_hpo_set:
		pred_hpo_set:
		hpo_dict (dict)
		hpo2ancestors {dict}: {hpo_code: {hpo_code1, ...}, ...}
	Returns:
		float
	"""
	hpo_dict = hpo_dict or get_hpo_dict()
	hpo2ancestors = hpo2ancestors or {hpo_code: get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	true_hpo_list, pred_hpo_list = list(true_hpo_set), list(pred_hpo_set)
	m = np.zeros((len(true_hpo_set), len(pred_hpo_set)), dtype=np.float64)
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			m[i, j] = set_precision(hpo2ancestors[true_hpo_list[i]], hpo2ancestors[pred_hpo_list[j]], ances_expand=False, hpo_dict=hpo_dict)
	return float(np.mean(np.max(m, axis=0)))   # mean of max value of each column


def set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict=None, hpo2ancestors=None):
	hpo_dict = hpo_dict or get_hpo_dict()
	hpo2ancestors = hpo2ancestors or {hpo_code: get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	true_hpo_list, pred_hpo_list = list(true_hpo_set), list(pred_hpo_set)
	m = np.zeros((len(true_hpo_set), len(pred_hpo_set)), dtype=np.float64)
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			m[i, j] = set_recall(hpo2ancestors[true_hpo_list[i]], hpo2ancestors[pred_hpo_list[j]], ances_expand=False, hpo_dict=hpo_dict)
	return float(np.mean(np.max(m, axis=1)))   # mean of max value of each row


def set_pair_f1(true_hpo_set, pred_hpo_set, hpo_dict=None, hpo2ancestors=None):
	hpo_dict = hpo_dict or get_hpo_dict()
	hpo2ancestors = hpo2ancestors or {hpo_code:get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	precision = set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict, hpo2ancestors=hpo2ancestors)
	recall = set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict, hpo2ancestors=hpo2ancestors)
	return 2 * (precision * recall) / (precision + recall)


def cal_recall_at_k(scores, elements, tgt_ele, k_list, has_sorted=False):
	"""
	Args:
		score_ele_list (list): [float, ...];
		elements (list): [ele, ...]; ele = e.g. hpo_code
		tgt_ele
		k_list (list): [k1, k2, ...]
		has_sorted (bool): whether the input scores has been sorted by score from large to small;
	Returns:
		list: length = len(k_list); element = 0 | 1
	"""
	tgt_rank = get_tgt_rank(scores, elements, tgt_ele, has_sorted)
	return [int(k >= tgt_rank) for k in k_list]


def get_tgt_rank(scores, elements, tgt_ele, k_list, has_sorted=False):
	"""
	Returns:
		int: range from 1 to len(scores)
	"""
	if not has_sorted:
		scores, elements = zip(*sorted(zip(scores, elements), key=lambda item: item[0], reverse=True))
	return elements.index(tgt_ele) + 1


def test():
	def equal(a, b, eps=1e-6):
		return abs(a - b) < eps

	hpo_dict = get_hpo_dict()
	old2new_hpo = hpo_dict.get
	true_hpo_set = {'HP:0004329', 'HP:0002558', 'HP:0010766'}
	pred_hpo_set = {'HP:0004329', 'HP:0002558', 'HP:0012374', 'HP:0000943'}
	print(get_all_ancestors_for_many(true_hpo_set, hpo_dict))   # 12
	print(get_all_ancestors_for_many(pred_hpo_set, hpo_dict))   # 13

	assert equal(set_precision(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 4)
	assert equal(set_recall(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 3)
	assert equal(set_f1(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 * ( (2 / 4) * (2 / 3) ) / (2 / 4 + 2 / 3))
	assert equal(set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 5)

	assert equal(set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 13)
	assert equal(set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 12)
	assert equal(set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 2 * ((11 / 13) * (11 / 12)) / (11 / 13 + 11 / 12))
	assert equal(set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 14)

	assert equal(set_pair_precision(true_hpo_set, pred_hpo_set), (1 + 1 + 1 + 3/5) / 4)
	assert equal(set_pair_recall(true_hpo_set, pred_hpo_set), (1 + 1 + 3 / 4) / 3)

	print(set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))
	print(set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))
	print(set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))

	print(set_pair_precision(true_hpo_set, pred_hpo_set))
	print(set_pair_recall(true_hpo_set, pred_hpo_set))
	print(set_pair_f1(true_hpo_set, pred_hpo_set))


def get_hpo_set(field_info, fields=None):
	fields = fields or ['现病史', '入院情况', '既往史', '家族史', '出院诊断']
	ret_hpo_list = []
	for field in fields:
		field_hpos = [tag_item['HPO_CODE'] for tag_item in field_info[field]['ENTITY_LIST']]
		ret_hpo_list.extend(field_hpos)
	return set(ret_hpo_list)


def cal_metrics(true_folder, pred_folder, pids):
	"""
	Args:
		true_folder (str)
		pred_folder (str)
		pids (list)
	Returns:
		dict: {SCORE_NAME: float}
	"""
	def get_pid2json(json_list):
		ret_dict = {}
		for json_path in json_list:
			pid = os.path.splitext(os.path.split(json_path)[1])[0]
			ret_dict[pid] = json_path
		return ret_dict
	true_pid2json = get_pid2json(get_file_list(true_folder, lambda p: p.endswith('.json')))
	pred_pid2json = get_pid2json(get_file_list(pred_folder, lambda p: p.endswith('.json')))
	hpo_dict = get_hpo_dict()

	result_dict = {}
	for pid in pids:
		true_hpo_set = get_hpo_set(json.load(open(true_pid2json[pid])))
		pred_hpo_set = get_hpo_set(json.load(open(pred_pid2json[pid])))
		dict_list_add('SET_EXPAND_RECALL', set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_EXPAND_PRECISION', set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_EXPAND_F1', set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_EXPAND_JACCARD', set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), result_dict)

		dict_list_add('SET_RECALL', set_recall(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_PRECISION', set_precision(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_F1', set_f1(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), result_dict)
		dict_list_add('SET_JACCARD', set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), result_dict)

		dict_list_add('SET_PAIR_RECALL', set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict),result_dict)
		dict_list_add('SET_PAIR_PRECISION', set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict),result_dict)
		dict_list_add('SET_PAIR_F1', set_pair_f1(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict), result_dict)

	for key, score_list in result_dict.items():
		result_dict[key] = np.mean(result_dict[key])

	return result_dict


def cal_metrics_for_many(true_folder, pred_folders, marks, pids, csv_path):
	df = pd.DataFrame()
	for pred_folder in pred_folders:
		result_dict = cal_metrics(true_folder, pred_folder, pids)
		df = df.append([result_dict])
	df.index = marks
	df.to_excel(csv_path)


if __name__ == '__main__':
	val_pids = json.load(open(os.path.join(DATA_PATH, 'raw', 'MER', 'PUMC', 'val_test_split', 'val_pids.json')))
	test_pids = json.load(open(os.path.join(DATA_PATH, 'raw', 'MER', 'PUMC', 'val_test_split', 'test_pids.json')))

	true_folder = os.path.join(DATA_PATH, 'raw', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance')
	pred_folders = [
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-ExactTermMatcher-CHPO'),
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-BagTermMatcher-CHPO'),
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-ExactTermMatcher-CHPO_MANUAL'),
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-BagTermMatcher-CHPO_MANUAL'),
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-ExactTermMatcher-CHPO_MANUAL_BG'),
		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'MaxInvTextSearcher-BagTermMatcher-CHPO_MANUAL_BG'),

		os.path.join(DATA_PATH, 'preprocess', 'patient', 'MER', 'PUMC', 'case87-doc-hy-strict-enhance', 'doc-bert'),
	]
	marks = [
		'CHPO-EXACT',
		'CHPO-BAG',
		'CHPO_MANUAL',
		'CHPO_MANUAL-BAG',
		'CHPO_MANUAL_BG',
		'CHPO_MANUAL_BG-BAG',
		'doc-bert',
	]
	cal_metrics_for_many(true_folder, pred_folders, marks, test_pids, os.path.join(RESULT_PATH, 'table', 'tag_test.xlsx'))

