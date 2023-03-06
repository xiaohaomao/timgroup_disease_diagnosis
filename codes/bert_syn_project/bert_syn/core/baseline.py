

import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from bert_syn.utils.utils import dict_list_extend, dict_list_add, cal_jaccard_sim_list, timer, jaccard_sim
from bert_syn.utils.constant import RESULT_PATH, DATA_PATH
from bert_syn.core.data_helper import HPOReader

class StandardAnalyzer(object):
	def __init__(self):
		self.current_type = self._get_type('')


	def split(self, s):
		s += '.'
		split_list = []
		begin = 0
		for i in range(len(s)):
			type = self._get_type(s[i])
			if self.current_type == 0:
				begin = i
			elif type > 2 or type != self.current_type:
				split_list.append(s[begin:i])
				begin = i
			self.current_type = type
		return split_list


	def _get_type(self, c):
		if self._is_num(c):			#连续
			return 1
		elif self._is_eng(c):		#连续
			return 2
		elif self._is_cns(c):		#非连续
			return 3
		else:
			return 0


	def _is_num(self, c):
		return '\u0030' <= c <='\u0039'


	def _is_eng(self, c):
		return ('\u0041' <= c <='\u005a') or ('\u0061' <= c <='\u007a')


	def _is_cns(self, c):
		return '\u4e00' <= c <= '\u9fff'


class TermMatcher(object):
	def __init__(self):
		pass


	def match(self, term):
		"""
		Returns:
			list: [(hpo_term, score), ...]
		"""
		raise NotImplementedError()


	def match_wrapper(self, term):
		return self.match(term), term


	def match_best(self, term):
		"""
		Returns:
			str: hpo_term
			float: score
		"""
		tgt_term_score_pairs = self.match(term)
		return max(tgt_term_score_pairs, key=lambda item: item[1])


	@timer
	def predict_and_save_csv(self, terms, csv_path, cpu_use=12, chunksize=50):
		os.makedirs(os.path.dirname(csv_path), exist_ok=True)
		row_dicts = []
		with Pool(cpu_use) as pool:
			for true_term_score_pairs, term in tqdm(pool.imap(self.match_wrapper, terms, chunksize=chunksize), total=len(terms), leave=False):
				row_dicts.extend([{'text_a':term, 'text_b':true_term, 'label':score} for true_term, score in true_term_score_pairs])
		pd.DataFrame(row_dicts, columns=['text_a', 'text_b', 'label']).to_csv(csv_path, index=False)


	def predict(self, terms, cpu_use=12, chunksize=50):
		"""
		Returns:
			list: [(term, true_term, score), ...]
		"""
		ret_samples = []
		with Pool(cpu_use) as pool:
			for true_term_score_pairs, term in tqdm(pool.imap(self.match_wrapper, terms, chunksize=chunksize), total=len(terms), leave=False):
				ret_samples.extend([(term, true_term, score) for true_term, score in true_term_score_pairs])
		return ret_samples


	@timer
	def predict_best(self, terms, cpu_use=12, chunksize=50):
		"""
		Returns:
			list: [(term, true_term, score), ...]
		"""
		with Pool(cpu_use) as pool:
			return pool.map(self.match_best, terms, chunksize=chunksize)


class ExactTermMatcher(TermMatcher):
	def __init__(self, syn_to_true_list):
		"""
		Args:
			syn_to_true_list (dict): e.g. {'巴氏征': ['巴彬斯基征']}
		"""
		super(ExactTermMatcher, self).__init__()
		self.syn_to_true_list = syn_to_true_list
		self.true_term_set = {true_term for syn, true_terms in syn_to_true_list.items() for true_term in true_terms}


	def match(self, term):
		"""
		Returns:
			list: [(hpo_term, score), ...]
		"""
		true_to_score = {true_term: 0.0 for true_term in self.true_term_set}
		if term in self.syn_to_true_list:
			for true_term in self.syn_to_true_list[term]:
				true_to_score[true_term] = 1.0
		return list(true_to_score.items())



class BagTermMatcher(TermMatcher):
	def __init__(self, syn_to_true_list):
		super(BagTermMatcher, self).__init__()
		self.analyzer = StandardAnalyzer()
		self.bag_string_to_true_list = self.get_bag_string_to_true_list(syn_to_true_list)
		self.true_term_set = {true_term for syn, true_terms in syn_to_true_list.items() for true_term in true_terms}


	def get_bag_string_to_true_list(self, syn_to_true_list):
		ret_dict = {}
		for syn_term, true_list in syn_to_true_list.items():
			dict_list_extend(self.get_bag_string(syn_term), true_list, ret_dict)
		return ret_dict


	def get_bag_string(self, term):
		return ''.join(sorted(self.analyzer.split(term)))


	def match(self, term):
		"""
		Returns:
			list: [(hpo_term, score), ...]
		"""
		true_to_score = {true_term:0.0 for true_term in self.true_term_set}
		bag_string = self.get_bag_string(term)
		if bag_string in self.bag_string_to_true_list:
			for true_term in self.bag_string_to_true_list[bag_string]:
				true_to_score[true_term] = 1.0
		return list(true_to_score.items())


class JaccardTermMatcher(TermMatcher):
	def __init__(self, syn_to_true_list):
		super(JaccardTermMatcher, self).__init__()
		self.syn_to_true_list = syn_to_true_list
		self.true_terms = list({true_term for syn, true_terms in syn_to_true_list.items() for true_term in true_terms})


	def match(self, term):
		"""
		Returns:
			list: [(hpo_term, score), ...]
		"""

		return [(true_term, jaccard_sim(set(term), set(true_term))) for true_term in self.true_terms ]


class DictSim(object):
	def __init__(self, name, hpo_to_syn_terms, match_type='exact'):
		"""
		Args:
			match_type (str): 'exact' | 'bag' | 'jaccard'
			hpo_to_syn_terms (dict): {hpo_code: [syn_term1, syn_term2, ...]}
		"""
		self.name = name
		self.syn_to_true_list = self.get_syn_to_true_list(hpo_to_syn_terms)
		if match_type == 'exact':
			self.matcher = ExactTermMatcher(self.syn_to_true_list)
		elif match_type == 'bag':
			self.matcher = BagTermMatcher(self.syn_to_true_list)
		elif match_type == 'jaccard':
			self.matcher = JaccardTermMatcher(self.syn_to_true_list)
		else:
			raise RuntimeError('Unknown match type: {}'.format(match_type))
		self.RESULT_SAVE_FOLDER = os.path.join(RESULT_PATH, self.name)
		os.makedirs(self.RESULT_SAVE_FOLDER, exist_ok=True)


	def get_syn_to_true_list(self, hpo_to_syn_terms):
		hpo_to_cns = HPOReader().get_hpo_to_cns()
		syn_to_true_list = {}
		for hpo, syn_terms in hpo_to_syn_terms.items():
			true_term = hpo_to_cns.get(hpo, '')
			if not true_term:
				continue
			for syn_term in syn_terms:
				dict_list_add(syn_term, true_term, syn_to_true_list)
		return syn_to_true_list


	def predict_and_save_csv(self, term_list_json, cpu_use=12, chunksize=50):
		save_csv = term_list_json.replace(os.path.join(DATA_PATH, 'preprocess', 'dataset'), self.RESULT_SAVE_FOLDER)
		save_csv = os.path.splitext(save_csv)[0] + '.csv'
		self.matcher.predict_and_save_csv(json.load(open(term_list_json)), save_csv, cpu_use=cpu_use, chunksize=chunksize)
		return save_csv


	def read_terms(self, terms):
		if isinstance(terms, str):
			if terms.endswith('.json'):
				return json.load(open(terms))
			elif terms.endswith('.txt'):
				return open(terms).read().strip().splitlines()
			else:
				assert False
		return terms


	def predict(self, terms, cpu_use=12, chunksize=50):
		"""
		Returns:
			list: [(term, hpo_term, score), ...]
		"""
		return self.matcher.predict(self.read_terms(terms), cpu_use=cpu_use, chunksize=chunksize)


	def predict_best(self, terms, cpu_use=12, chunksize=50):
		"""
		Returns:
			list: [(hpo_term, score), ...]
		"""
		return self.matcher.predict_best(self.read_terms(terms), cpu_use=cpu_use, chunksize=chunksize)


	def predict_best_hpos(self, terms, cpu_use=12, chunksize=50):
		"""
		Returns:
			list: [(hpo, score), ...]
		"""
		hpoterm_score_pairs = self.predict_best(terms, cpu_use, chunksize)
		cns_to_hpos = HPOReader().get_cns_to_hpo()
		ret = []
		for hpo_term, score in hpoterm_score_pairs:
			ret.append((cns_to_hpos))


if __name__ == '__main__':
	pass
