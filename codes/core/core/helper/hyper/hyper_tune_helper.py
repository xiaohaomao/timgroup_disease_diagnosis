

import numpy as np
import os
import heapq
import json
from copy import deepcopy
import random, scipy
import itertools

from core.utils.utils import flatten_dict, to_rank_scores, to_zscores, is_zero
from core.utils.constant import RESULT_PATH, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE, HYPER_TUNE_USER_SCORE
from core.utils.constant import HYPER_TUNE_AVE_SCORE, HYPER_TUNE_SUCC_Z_SCORE
from core.draw.simpledraw import simple_dot_plot


class HyperTuneHelper(object):
	def __init__(self, name=None, score_keys=None, score_weights=None, mode='a', save_folder=None):
		"""
		Args
			score_keys (list or None): e.g. [['DATASET1', 'DATASET2', ...], ['METRIC1', 'METRIC2', ...]] => score_dict = {'DATASET1': {'METRIC1': score, 'METRIC2': score}, ...}
			score_weights (list or None): e.g. [{'DATASET1': 0.5, 'DATASET2': 0.5, ...}, {'METRIC1': 0.5, 'METRIC2': 0.5, ...}]
			ave_score_keys (list or None): [['DATASET1', 'DATASET2', ...], ['METRIC1', 'METRIC2', ...]]
		"""
		self.KEY_SEP = '-'
		self.init_save_path(name, save_folder)
		self.init_history(mode)
		self.flt_score_weights = self.cal_flt_score_weights(score_keys, score_weights)
		self.flt_score_order = self.cal_flt_score_order(score_keys, self.flt_score_weights)
		self.flt_score_order_w = self.cal_flt_score_order_weight_sums(self.flt_score_order, self.flt_score_weights)


	def init_save_path(self, name=None, save_folder=None):
		if save_folder is None:
			assert name is not None
			self.HISTORY_FOLDER = os.path.join(RESULT_PATH, 'hyper_tune', '{}'.format(name))
		else:
			self.HISTORY_FOLDER = save_folder
		os.makedirs(self.HISTORY_FOLDER, exist_ok=True)

		self.HISTORY_JSON = os.path.join(self.HISTORY_FOLDER, 'history.json')
		self.FLT_SCORE_WEIGHTS = os.path.join(self.HISTORY_FOLDER, 'score_weights.json')
		self.FLT_SCORE_ORDER = os.path.join(self.HISTORY_FOLDER, 'score_order.json')


	def get_score_iteration_fig_path(self, sort_type):
		return os.path.join(self.HISTORY_FOLDER, 'ScoreWithIter-{}.png'.format(sort_type))


	def get_score_with_para_fig_path(self, key, sort_type):
		return os.path.join(self.HISTORY_FOLDER, '{}-{}.png'.format(key, sort_type))


	def get_sorted_history_json(self, sort_type):
		return os.path.join(self.HISTORY_FOLDER, 'SortedHistory-{}.json'.format(sort_type))


	def cal_flt_score_weights(self, score_keys, score_weights):
		if score_keys is None:
			assert os.path.exists(self.FLT_SCORE_WEIGHTS)
			return json.load(open(self.FLT_SCORE_WEIGHTS))
		if score_weights is None: # average weight
			w = 1.0
			for skeys in score_keys:
				w /= len(skeys)
			return {self.KEY_SEP.join(skeys): w for skeys in itertools.product(*score_keys)}
		ret = {}
		for skeys in itertools.product(*score_keys): # given weight
			w = 1.0
			for i, k in enumerate(skeys):
				w *= score_weights[i][k]
			ret[self.KEY_SEP.join(skeys)] = w
		return ret


	def cal_flt_score_order(self, score_keys, flt_score_weight):
		"""
		Args:
			score_keys (list): [['DATASET1', 'DATASET2', ...], ['METRIC1', 'METRIC2', ...]]
			flt_score_weight: {'RAMEDIS-Mic.RankMedian': weight, ...}
		Returns:
			list: [(DATASET1-METRIC1, DATASET2-METRIC1, ...), (DATASET1-METRIC2, DATASET2-METRIC2, ...), ...]
		"""
		if score_keys is None:
			assert os.path.exists(self.FLT_SCORE_ORDER)
			return json.load(open(self.FLT_SCORE_ORDER))
		assert len(score_keys) == 2
		ret_list = []
		for metric_name in score_keys[1]:
			flt_score_names = [dname+self.KEY_SEP+metric_name for dname in score_keys[0]]
			flt_score_names = [flt_score_name for flt_score_name in flt_score_names if not is_zero(flt_score_weight[flt_score_name])]
			if flt_score_names:
				ret_list.append(flt_score_names)
		return ret_list


	def cal_flt_score_order_weight_sums(self, flt_score_order, flt_score_weights):
		"""
		Returns:
			list: [{flt_score_names: weight}, ...], length = len(flt_score_order)
		"""
		ret_list = []
		for flt_score_names in flt_score_order:
			w_dict = {k: flt_score_weights[k] for k in flt_score_names}
			weights_sum = sum(w_dict.values())
			ret_list.append({k: w / weights_sum for k, w in w_dict.items()})
		return ret_list


	def set_flt_score_weights(self, score_keys, score_weights):
		self.flt_score_weights = self.cal_flt_score_weights(score_keys, score_weights)


	def init_history(self, mode):
		if mode == 'a':
			self.load_history()
		elif mode =='w':
			self.history = []
		else:
			assert False


	def get_H(self, i):
		return self.history[i]


	def get_history_length(self):
		return len(self.history)


	def get_score_key(self, skeys):
		return self.KEY_SEP.join(skeys)


	def add(self, para_dict, score_item=None, score_dict=None):
		"""
		Args:
			para_dict (dict): {key: value, ...}
			score_item (float or list or tuple or None): score | score_list | scoreTuple
			score_dict (dict or None):
		"""
		assert score_item is not None or score_dict is not None
		d = {'PARAMETER': para_dict}
		if score_item is not None:
			d['SCORE_ITEM'] = score_item
		if score_dict is not None:
			d['FLT_SCORE_DICT'] = flatten_dict(score_dict)
		self.history.append(d)


	def add_many(self, history):
		"""
		Args:
			history (list): [{'SCORE_ITEM': score_item, 'PARAMETER': para_dict, 'FLT_SCORE_DICT': score_dict}, ...]
		"""
		self.history.extend(history)


	def history_filter(self, history, keep_func=None):
		if keep_func is None:
			return history
		return [h for h in history if keep_func(h)]


	def get_enrich_history(self, sort_type):
		history = deepcopy(self.history)
		sort_score_list = self.cal_history_sort_score(sort_type)
		for h, d in zip(history, sort_score_list):
			h.update(d)
		return history


	def get_topk(self, topk=10, keep_func = None, sort_type=HYPER_TUNE_Z_SCORE):
		history = self.get_enrich_history(sort_type)
		return heapq.nlargest(topk, self.history_filter(history, keep_func), key=lambda item: item['SCORE'])


	def get_sorted_history(self, keep_func=None, sort_type=HYPER_TUNE_Z_SCORE):
		history = self.get_enrich_history(sort_type)
		return sorted(self.history_filter(history, keep_func), key=lambda h: h['SCORE'], reverse=True)


	def get_arg_best(self, keep_func=None, sort_type=HYPER_TUNE_Z_SCORE):
		history = self.get_enrich_history(sort_type)
		if isinstance(history[0]['SCORE'], tuple):
			max_score, best_rank = history[0]['SCORE'], 0
			for i, h in enumerate(self.history[1:]):
				if h['SCORE'] > max_score:
					max_score, best_rank = h['SCORE'], i
			return best_rank
		return np.argmax([h['SCORE'] for h in self.history_filter(history, keep_func)])


	def get_best(self, keep_func=None, sort_type=HYPER_TUNE_Z_SCORE):
		if len(self.history) == 0:
			return {}
		return self.history[self.get_arg_best(keep_func, sort_type)]


	def get_best_para(self, keep_func=None, sort_type=HYPER_TUNE_Z_SCORE):
		return self.get_best(keep_func, sort_type)['PARAMETER']


	def get_para_history(self):
		return [h['PARAMETER'] for h in self.history]


	def write_as_str(self, obj):
		return json.dumps(obj, indent=2, ensure_ascii=False)


	def save_history(self):
		json.dump(self.history, open(self.HISTORY_JSON, 'w'), indent=2)
		json.dump(self.get_sorted_history(sort_type=HYPER_TUNE_RANK_SCORE), open(self.get_sorted_history_json(HYPER_TUNE_RANK_SCORE), 'w'), indent=2)
		json.dump(self.get_sorted_history(sort_type=HYPER_TUNE_Z_SCORE), open(self.get_sorted_history_json(HYPER_TUNE_Z_SCORE), 'w'), indent=2)
		json.dump(self.get_sorted_history(sort_type=HYPER_TUNE_SUCC_Z_SCORE), open(self.get_sorted_history_json(HYPER_TUNE_SUCC_Z_SCORE), 'w'), indent=2)
		json.dump(self.get_sorted_history(sort_type=HYPER_TUNE_AVE_SCORE), open(self.get_sorted_history_json(HYPER_TUNE_AVE_SCORE), 'w'), indent=2)
		json.dump(self.flt_score_weights, open(self.FLT_SCORE_WEIGHTS, 'w'), indent=2)
		json.dump(self.flt_score_order, open(self.FLT_SCORE_ORDER, 'w'), indent=2)


	def load_history(self, create=True):
		if os.path.exists(self.HISTORY_JSON):
			self.history = json.load(open(self.HISTORY_JSON))
		elif create:
			self.history = []
		else:
			assert False


	def cal_history_sort_score(self, sort_type):
		if sort_type == HYPER_TUNE_USER_SCORE:
			return self.cal_history_user_score()
		elif sort_type == HYPER_TUNE_RANK_SCORE:
			return self.cal_history_rank_score()
		elif sort_type == HYPER_TUNE_Z_SCORE:
			return self.cal_history_zscore()
		elif sort_type == HYPER_TUNE_SUCC_Z_SCORE:
			return self.cal_history_succ_zscore()
		elif sort_type == HYPER_TUNE_AVE_SCORE:
			return self.cal_history_ave_score()
		else:
			raise RuntimeError('Unknown sort_type: {}'.format(sort_type))


	def cal_history_user_score(self):
		"""
		Returns:
			list: [{'SCORE': float}, ...]
		"""
		return [{'SCORE': h['SCORE_ITEM']} for h in self.history]


	def cal_history_rank_score(self):
		"""
		Returns:
			list: [{'RANK_SCORE': {}, 'SCORE': float}, ...]
		"""
		assert self.flt_score_weights is not None
		ret = [{'RANK_SCORE': {}, 'SCORE': 0.0} for h in self.history]
		for flt_score_key in self.flt_score_weights:
			score_list = to_rank_scores([h['FLT_SCORE_DICT'][flt_score_key] for h in self.history])
			for i, score in enumerate(score_list):
				ret[i]['RANK_SCORE'][flt_score_key] = score
		for item in ret:
			item['SCORE'] = sum([self.flt_score_weights[k] * item['RANK_SCORE'][k] for k in self.flt_score_weights])
		return ret


	def cal_history_zscore(self):
		"""
		Returns:
			list: [{'Z_SCORE': {}, 'SCORE': float}, ...]
		"""
		assert self.flt_score_weights is not None
		ret = [{'Z_SCORE':{}, 'SCORE':0.0} for h in self.history]
		for flt_score_key in self.flt_score_weights:
			score_list = to_zscores([h['FLT_SCORE_DICT'][flt_score_key] for h in self.history])
			for i, score in enumerate(score_list):
				ret[i]['Z_SCORE'][flt_score_key] = score
		for item in ret:
			item['SCORE'] = sum([self.flt_score_weights[k] * item['Z_SCORE'][k] for k in self.flt_score_weights])
		return ret


	def cal_history_succ_zscore(self):
		"""
		Returns:
			list: [{'Z_SCORE': {}, 'SCORE': (score1, score2, ...)}, ...]
		"""
		z_score_dicts = [{} for h in self.history]
		for flt_score_key in self.flt_score_weights:
			score_list = to_zscores([h['FLT_SCORE_DICT'][flt_score_key] for h in self.history])
			for i, score in enumerate(score_list):
				z_score_dicts[i][flt_score_key] = score
		ret = []
		for z_score_dict in z_score_dicts:
			score_tuple = tuple( sum([flt_score_weights[k] * z_score_dict[k] for k in flt_score_names])
				for flt_score_names, flt_score_weights in zip(self.flt_score_order, self.flt_score_order_w))
			ret.append({'Z_SCORE': z_score_dict, 'SCORE': score_tuple})
		return ret


	def cal_history_ave_score(self):
		"""
		Returns:
			list: [{'SCORE': (score1, score2, ...)}, ...]
		"""
		ret = []
		for h in self.history:
			score_tuple = tuple( sum([flt_score_weights[k] * h['FLT_SCORE_DICT'][k] for k in flt_score_names])
				for flt_score_names, flt_score_weights in zip(self.flt_score_order, self.flt_score_order_w))
			ret.append({'SCORE': score_tuple})
		return ret


	def score_item_to_score(self, score_item):
		if isinstance(score_item, list) or isinstance(score_item, tuple):
			return score_item[0]
		return score_item


	def draw_score_with_iteration(self, figpath=None, shuffle=False, filter=None, sort_type=HYPER_TUNE_RANK_SCORE):
		if figpath is None:
			figpath = self.get_score_iteration_fig_path(sort_type)
		history = self.get_enrich_history(sort_type)
		if shuffle:
			random.shuffle(history)
		if filter is not None:
			history = [h for h in history if filter(h)]
		simple_dot_plot(
			figpath,
			list(range(len(history))),
			[self.score_item_to_score(h['SCORE']) for h in history],
			'Iteration', sort_type
		)


	def draw_score_with_para(self, key, figpath=None, sort_type=HYPER_TUNE_RANK_SCORE):
		if figpath is None:
			figpath = self.get_score_with_para_fig_path(key, sort_type)
		history = self.get_enrich_history(sort_type)
		history = [h for h in history if key in h['PARAMETER']]
		simple_dot_plot(
			figpath,
			[(str(h['PARAMETER'][key]) if isinstance(h['PARAMETER'][key], tuple) or isinstance(h['PARAMETER'][key], list) else h['PARAMETER'][key]) for h in history],
			[self.score_item_to_score(h['SCORE']) for h in history],
			key, sort_type
		)


def resort_history(save_folder, score_keys, score_weights):
	HyperTuneHelper(score_keys=score_keys, score_weights=score_weights, save_folder=save_folder).save_history()


def combine_history(old_save_folders, new_save_folder, score_keys=None, score_weights=None, keep_func=None):
	hth1 = HyperTuneHelper(save_folder=old_save_folders[0], score_keys=score_keys, score_weights=score_weights)
	for folder in old_save_folders[1:]:
		hth2 = HyperTuneHelper(save_folder=folder)
		assert hth1.flt_score_order == hth2.flt_score_order
		hth1.add_many(hth2.history)
	if keep_func is not None:
		hth1.history = [h for h in hth1.history if keep_func(h)]
	hth1.init_save_path(save_folder=new_save_folder)
	hth1.save_history()



if __name__ == '__main__':
	pass


