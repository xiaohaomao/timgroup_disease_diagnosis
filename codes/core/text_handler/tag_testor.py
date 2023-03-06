

import os
import numpy as np
import json

from core.utils.constant import RESULT_PATH
from core.utils.utils import get_all_ancestors_for_many, get_all_ancestors, delete_redundacy, dict_list_add
from core.reader import HPOReader, HPOFilterDatasetReader


def set_precision(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=True):
	"""
	Args:
		true_hpo_set (set): {hpo_code1, ...}
		pred_hpo_set (set): {hpo_code1, ...}
		ances_expand (bool)
		hpo_dict (dict)
	Returns:
		float
	"""
	if len(pred_hpo_set) == 0:
		return 0.0
	if ances_expand:
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(pred_hpo_set)


def set_recall(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=True):
	if ances_expand:
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(true_hpo_set)


def set_f1(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=True):
	if ances_expand:
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	precision = set_precision(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False)
	recall = set_recall(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False)
	if precision + recall < 1e-6:
		return 0.0
	return 2 * (precision * recall) / (precision + recall)


def set_jaccard(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=True):
	if ances_expand:
		true_hpo_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
		pred_hpo_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)
	return len(true_hpo_set & pred_hpo_set) / len(true_hpo_set | pred_hpo_set)


def set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict, hpo2ancestors=None):
	"""
	Args:
		true_hpo_set:
		pred_hpo_set:
		hpo_dict (dict)
		hpo2ancestors {dict}: {hpo_code: {hpo_code1, ...}, ...}
	Returns:
		float
	"""
	hpo2ancestors = hpo2ancestors or {hpo_code: get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	true_hpo_list, pred_hpo_list = list(true_hpo_set), list(pred_hpo_set)
	m = np.zeros((len(true_hpo_set), len(pred_hpo_set)), dtype=np.float64)
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			m[i, j] = set_precision(hpo2ancestors[true_hpo_list[i]], hpo2ancestors[pred_hpo_list[j]], ances_expand=False, hpo_dict=hpo_dict)
	return float(np.mean(np.max(m, axis=0)))   # mean of max value of each column


def set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict, hpo2ancestors=None):
	hpo2ancestors = hpo2ancestors or {hpo_code: get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	true_hpo_list, pred_hpo_list = list(true_hpo_set), list(pred_hpo_set)
	m = np.zeros((len(true_hpo_set), len(pred_hpo_set)), dtype=np.float64)
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			m[i, j] = set_recall(hpo2ancestors[true_hpo_list[i]], hpo2ancestors[pred_hpo_list[j]], ances_expand=False, hpo_dict=hpo_dict)
	return float(np.mean(np.max(m, axis=1)))   # mean of max value of each row


def set_pair_f1(true_hpo_set, pred_hpo_set, hpo_dict, hpo2ancestors=None):
	hpo2ancestors = hpo2ancestors or {hpo_code:get_all_ancestors(hpo_code, hpo_dict) for hpo_code in true_hpo_set | pred_hpo_set}
	precision = set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict, hpo2ancestors=hpo2ancestors)
	recall = set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict=hpo_dict, hpo2ancestors=hpo2ancestors)
	return 2 * (precision * recall) / (precision + recall)


def cal_recall_at_k(scores, elements, tgt_ele, k_list, has_sorted=False):
	"""
	Args:
		scores (list): [float, ...];
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


class TagTestor(object):
	def __init__(self, hpo_reader=HPOReader()):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterReader):
		"""
		self.hpo_reader = hpo_reader
		self.metirc_names = [
			'SET_ANCES_F1',
			'SET_ANCES_RECALL',
			'SET_ANCES_PRECISION',
			'SET_ANCES_JACCARD',

			'SET_EXACT_F1',
			'SET_EXACT_RECALL',
			'SET_EXACT_PRECISION',
			'SET_EXACT_JACCARD'
		]
		self.field_names = ['现病史', '入院情况', '出院诊断', '既往史']
		self.RESULT_PATH = os.path.join(RESULT_PATH, 'tag')
		os.makedirs(self.RESULT_PATH, exist_ok=True)


	def process_field_to_info_json(self, json_path, field_names=None):
		"""
		Args:
			json_path (str)
			ances_expand (bool)
		Returns:
			set: {hpo1, hpo2, ...}
		"""
		field_to_info = json.load(open(json_path))
		hpo_set = set()
		for field_name in field_names:
			info = field_to_info[field_name]
			for entity_item in info['ENTITY_LIST']:
				if len(entity_item['HPO_CODE'].strip()) > 0:
					hpo_set.add(entity_item['HPO_CODE'])

		hpo_dict = self.hpo_reader.get_hpo_dict()
		chpo_dict = self.hpo_reader.get_chpo_dict()
		hpo_old_to_new = self.hpo_reader.get_old_map_new_hpo_dict()
		ret_hpo_set = set()
		for hpo in hpo_set:
			if hpo in hpo_dict:
				ret_hpo_set.add(hpo)
			elif hpo in hpo_old_to_new:
				ret_hpo_set.add(hpo_old_to_new[hpo])
			else:
				print('Drop hpo: {} ({})'.format(hpo, chpo_dict[hpo].get('CNS_NAME', '') or hpo_dict.get('ENG_NAME', '')))
		return set(delete_redundacy(ret_hpo_set, hpo_dict))


	def cal_metrics(self, true_folder, pred_folder, json_names, metric_names=None, field_names=None):
		"""
		Args:
			true_folder (str):
			pred_folder (str):
			file_names (list): ['0001.json', ...]
		Returns:
			dict: {METRIC_NAME: float}
		"""
		metric_names = metric_names or self.metirc_names
		field_names = field_names or self.field_names
		hpo_dict = self.hpo_reader.get_hpo_dict()
		metric_to_vlist = {}
		for json_file in json_names:
			true_hpo_set = self.process_field_to_info_json(os.path.join(true_folder, json_file), field_names)
			true_hpo_ances_set = get_all_ancestors_for_many(true_hpo_set, hpo_dict)
			pred_hpo_set = self.process_field_to_info_json(os.path.join(pred_folder, json_file), field_names)
			pred_hpo_ances_set = get_all_ancestors_for_many(pred_hpo_set, hpo_dict)

			if 'SET_ANCES_F1' in metric_names:
				dict_list_add('SET_ANCES_F1', set_f1(true_hpo_ances_set, pred_hpo_ances_set,hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_ANCES_RECALL' in metric_names:
				dict_list_add('SET_ANCES_RECALL', set_recall(true_hpo_ances_set, pred_hpo_ances_set, hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_ANCES_PRECISION' in metric_names:
				dict_list_add('SET_ANCES_PRECISION', set_precision(true_hpo_ances_set, pred_hpo_ances_set, hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_ANCES_JACCARD' in metric_names:
				dict_list_add('SET_ANCES_JACCARD', set_jaccard(true_hpo_ances_set, pred_hpo_ances_set, hpo_dict, ances_expand=False), metric_to_vlist)

			if 'SET_EXACT_F1' in metric_names:
				dict_list_add('SET_EXACT_F1', set_f1(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_EXACT_RECALL' in metric_names:
				dict_list_add('SET_EXACT_RECALL', set_recall(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_EXACT_PRECISION' in metric_names:
				dict_list_add('SET_EXACT_PRECISION', set_precision(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False), metric_to_vlist)
			if 'SET_EXACT_JACCARD' in metric_names:
				dict_list_add('SET_EXACT_JACCARD', set_jaccard(true_hpo_set, pred_hpo_set, hpo_dict, ances_expand=False), metric_to_vlist)

		return {metric_name: float(np.mean(vlist)) for metric_name, vlist in metric_to_vlist.items()}


	def cal_metrics_and_save(self, true_folder, pred_folder, json_names, metric_names=None, field_names=None, save_json=None):
		save_json = save_json or os.path.join(self.RESULT_PATH, os.path.split(pred_folder)[1]+'.json')
		metric_dict = self.cal_metrics(true_folder, pred_folder, json_names, metric_names=metric_names, field_names=field_names)
		print(json.dumps(metric_dict, indent=2))
		json.dump(metric_dict, open(save_json, 'w'), indent=2)


def test():
	def equal(a, b, eps=1e-6):
		return abs(a - b) < eps

	hpo_dict = HPOReader().get_hpo_dict()
	true_hpo_set = {'HP:0004329', 'HP:0002558', 'HP:0010766'}
	pred_hpo_set = {'HP:0004329', 'HP:0002558', 'HP:0012374', 'HP:0000943'}
	print(get_all_ancestors_for_many(true_hpo_set, hpo_dict))  # 12
	print(get_all_ancestors_for_many(pred_hpo_set, hpo_dict))  # 13

	assert equal(set_precision(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 4)
	assert equal(set_recall(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 3)
	assert equal(set_f1(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict),
		2 * ((2 / 4) * (2 / 3)) / (2 / 4 + 2 / 3))
	assert equal(set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=False, hpo_dict=hpo_dict), 2 / 5)

	assert equal(set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 13)
	assert equal(set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 12)
	assert equal(set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict),
		2 * ((11 / 13) * (11 / 12)) / (11 / 13 + 11 / 12))
	assert equal(set_jaccard(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict), 11 / 14)

	assert equal(set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict), (1 + 1 + 1 + 3 / 5) / 4)
	assert equal(set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict), (1 + 1 + 3 / 4) / 3)

	print(set_precision(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))
	print(set_recall(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))
	print(set_f1(true_hpo_set, pred_hpo_set, ances_expand=True, hpo_dict=hpo_dict))

	print(set_pair_precision(true_hpo_set, pred_hpo_set, hpo_dict))
	print(set_pair_recall(true_hpo_set, pred_hpo_set, hpo_dict))
	print(set_pair_f1(true_hpo_set, pred_hpo_set, hpo_dict))


if __name__ == '__main__':
	from core.utils.constant import DATA_PATH
	tt = TagTestor()

	true_folder = os.path.join(DATA_PATH, 'raw', 'PUMC', 'case87-doc-hy-strict-enhance')
	pids = json.load(open(os.path.join(DATA_PATH, 'raw', 'PUMC', 'val_test_split', 'test_pids.json')))
	json_names = [f'{pid}.json' for pid in pids]

	base_folder = '/home/yhuang/RareDisease/bert_syn_project/data/preprocess/pumc_87'


	pred_folder = os.path.join(base_folder, 'dict_bert-albertTinyDDMLSim')


	tt.cal_metrics_and_save(true_folder, pred_folder, json_names)
