import os
import numpy as np
import heapq
from tqdm import tqdm
import pickle
import warnings

from core.predict.model import Model
from core.utils.utils import get_all_ancestors_for_many, item_list_to_rank_list, timer, list_find
from core.utils.constant import PHELIST_REDUCE, MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.reader.hpo_reader import HPOReader
from multiprocessing import Pool


class BayesNetModel(Model):
	def __init__(self, hpo_reader, alpha=0.5, cond_type='max', init_cpu=12, mode=PREDICT_MODE, model_name=None, init_para=True):
		"""
		Args:
			cond_type (str): 'max' | 'ind'
		"""
		super(BayesNetModel, self).__init__()
		self.name = 'BayesNetModel' if model_name is None else model_name
		self.MODEL_SAVE_FOLDER = MODEL_PATH + os.sep + 'BayesNetModel'
		os.makedirs(self.MODEL_SAVE_FOLDER, exist_ok=True)
		self.UNBIAS_SAVE_PKL = self.MODEL_SAVE_FOLDER + os.sep + '{}_unbias.json'.format(self.name)
		self.DISBIAS_SAVE_PKL = self.MODEL_SAVE_FOLDER + os.sep + '{}_dis_bias.json'.format(self.name)

		self.hpo_reader = hpo_reader
		self.DIS_CODE_NUMBER = hpo_reader.get_dis_num()
		self.HPO_CODE_NUMBER = hpo_reader.get_hpo_num()
		self.dis_list = hpo_reader.get_dis_list()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()

		self.alpha = alpha
		self.cond_type = cond_type
		self.unbias = None
		self.dis_bias = None
		self.init_cpu = init_cpu
		self.EPS = 1e-9

		if init_para and mode == PREDICT_MODE:
			model.load()


	@timer
	def train(self, save_model=False):
		self.unbias = self.get_hpo_unbias_log_prob()
		self.dis_bias = self.get_dis_bias_log_prob()
		if save_model:
			self.save()


	def save(self):
		pickle.dump(self.unbias, open(self.UNBIAS_SAVE_PKL, 'wb'))
		pickle.dump(self.dis_bias, open(self.DISBIAS_SAVE_PKL, 'wb'))


	def load(self):
		self.unbias = pickle.load(open(self.UNBIAS_SAVE_PKL, 'rb'))
		self.dis_bias = pickle.load(open(self.DISBIAS_SAVE_PKL, 'rb'))


	def get_hpo_unbias_log_prob(self):
		"""
		Returns:
			dict: {hpo: {child: log_prob}}
		"""
		ret_dict ={}
		for hpo, info in self.hpo_reader.get_hpo_int_dict().items():
			children = info.get('CHILD', [])
			if len(children) == 0:
				continue
			log_prob = -np.log(len(children))
			ret_dict[hpo] = {child:log_prob for child in children}
		return ret_dict


	def get_dis_bias_log_prob(self):
		dis_bias = [0] * self.HPO_CODE_NUMBER
		with Pool(self.init_cpu) as pool:
			for bias, dis_int in tqdm(
					pool.imap_unordered(self.get_hpo_bias_log_prob_multi_wrap, range(self.DIS_CODE_NUMBER), chunksize=400),
					total=self.DIS_CODE_NUMBER, leave=False):
				dis_bias[dis_int] = bias
		return dis_bias


	def hpo_int_list_to_child_dict(self, hpo_int_list):
		"""
		Returns:
			dict: {hpo_int: [child, ...]}
		"""
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		ancestors = get_all_ancestors_for_many(hpo_int_list, hpo_int_dict)
		return {hpo: [child for child in hpo_int_dict[hpo].get('CHILD', []) if child in ancestors] for hpo in ancestors}


	def hpo_int_list_to_parent_dict(self, hpo_int_list):
		"""
		Returns:
			dict: {hpo_int: [parent, ...]}
		"""
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		ancestors = get_all_ancestors_for_many(hpo_int_list, hpo_int_dict)
		return {hpo: [p for p in hpo_int_dict[hpo].get('IS_A', []) if p in ancestors] for hpo in ancestors}


	def get_hpo_bias_log_prob_multi_wrap(self, dis_int):
		return self.get_hpo_bias_log_prob(dis_int), dis_int


	def get_hpo_bias_log_prob(self, dis_int):
		"""
		Returns:
			dict: {hpo_int: {child: log_prob}}
		"""
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		dhpo_int_list = self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_REDUCE)[dis_int]
		dhpo_to_childs = self.hpo_int_list_to_child_dict(dhpo_int_list)
		ret_dict = {}
		for pHPO, dchilds in dhpo_to_childs.items():
			if len(dchilds) == 0:
				continue
			ret_dict[pHPO] = {}
			dchilds = set(dchilds); dchild_num = len(dchilds)
			childs = hpo_int_dict[pHPO].get('CHILD', []); child_num = len(childs)
			for child in childs:
				if child in dchilds:
					ret_dict[pHPO][child] = np.log(1 + self.alpha) - np.log( dchild_num + child_num * self.alpha)
				else:
					ret_dict[pHPO][child] = np.log(0 + self.alpha) - np.log(dchild_num + child_num * self.alpha)
		return ret_dict


	def cal_cond_log_prob(self, hpo, parents, bias, unbias):
		if len(parents) == 0:
			assert hpo == 0
			return 0.0
		condprobs = [bias.get(p, {}).get(hpo, None) or unbias[p][hpo] for p in parents]
		return self.combine_cond_probs(condprobs, self.cond_type)


	def combine_cond_probs(self, condprobs, cond_type='max'):
		if len(condprobs) == 1:
			return condprobs[0]
		if cond_type == 'max':
			return max(condprobs)
		if cond_type == 'ind':
			logp = np.array(condprobs)
			tmp = 1 - np.exp(logp); tmp[tmp <= 0] = self.EPS
			tmp = 1 - np.exp(np.log(tmp).sum()); tmp = self.EPS if tmp <= 0 else tmp
			return np.log(tmp)
		assert False


	def cal_generate_prob(self, phpo_to_parents, dis_int):
		return np.sum([self.cal_cond_log_prob(hpo, parents, self.dis_bias[dis_int], self.unbias) for hpo, parents in phpo_to_parents.items()])


	def cal_score(self, phe_list):
		phe_int_list = item_list_to_rank_list(phe_list, self.hpo_reader.get_hpo_map_rank())
		phpo_to_parents = self.hpo_int_list_to_parent_dict(phe_int_list)
		return [self.cal_generate_prob(phpo_to_parents, i) for i in range(self.DIS_CODE_NUMBER)]


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		score_vec = self.cal_score(phe_list)
		assert np.sum(np.isnan(score_vec)) == 0
		return score_vec


	def score_vec_to_result(self, score_vec, topk):
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1])  # [(dis_code, score), ...], shape=(dis_num, )


if __name__ == '__main__':
	model = BayesNetModel(alpha=0.0001, cond_type='ind', mode=TRAIN_MODE)
	model.train()

