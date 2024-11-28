"""
"""

import re
from copy import deepcopy

from core.reader.hpo_reader import HPOReader
from core.analyzer.standard_analyzer import StandardAnalyzer
from core.utils.utils import reverse_dict_list, get_all_descendents_with_dist
from core.utils.constant import ROOT_HPO_CODE
from core.reader.umls_reader import UMLSReader
from core.text_handler.syn_generator import SynGenerator


class TermMatcher(object):
	def __init__(self, hpo_to_golds=None, hpo_to_gold_ps=None):
		"""
		Args:
			hpo_to_golds (dict): {hpo_code: [cns_term, ...]}; hpo_code can be empty
			hpo_to_gold_ps (dict): {hpo_code: [pattern, ...]}
		"""
		self.hpo_reader = HPOReader()
		self.set_hpo_to_golds(hpo_to_golds)
		self.set_hpo_to_gold_cps(hpo_to_gold_ps)


	def match(self, term):
		"""
		Args:
			term (str)
		Returns:
			list: [hpo_code, ...]
			list: [score, ...]
		"""
		raise NotImplementedError


	def set_hpo_to_golds(self, hpo_to_golds):
		hpo_to_golds = {} if hpo_to_golds is None else hpo_to_golds
		self.gold_to_hpos = self.get_term_to_hpos(hpo_to_golds)


	def set_hpo_to_gold_cps(self, hpo_to_gold_ps):
		hpo_to_gold_ps = {} if hpo_to_gold_ps is None else hpo_to_gold_ps

		# # DEBUG
		# for hpo, p_list in hpo_to_gold_ps.items():
		# 	print(hpo, p_list)
		# 	print(re.compile('^'+'$|^'.join(p_list)+'$'))

		self.hpo_to_gold_cp = {hpo: re.compile('^'+'$|^'.join(p_list)+'$') for hpo, p_list in hpo_to_gold_ps.items()} # cp: compile pattern; p: pattern str
		# self.combine_gold_p = re.compile('^'+'|'.join( ['|'.join(p_list) for p_list in hpo_to_gold_ps.values()])+'$')


	def match_gold(self, term):
		"""
		Returns:
			bool: is gold term
			list: [hpo_code, ...]
			list: [score, ...]
		"""
		is_gold, ret_match_list, ret_score_list = False, [], []
		if term in self.gold_to_hpos:
			is_gold = True; ret_match_list.extend(self.gold_to_hpos[term]); ret_score_list.extend([1]*len(self.gold_to_hpos[term]))
		for hpo, cp in self.hpo_to_gold_cp.items():
			if cp.match(term):
				if hpo is None:
					return True, [], []
				is_gold = True; ret_match_list.append(hpo); ret_score_list.append(1)
		return is_gold, ret_match_list, ret_score_list


	def get_max_term_length(self):
		raise NotImplementedError


	def get_term_to_hpos(self, hpo_to_terms):
		"""
		Returns:
			{hpo_code1: [cns_term, ...], hpo_code2: []}
		"""
		none_hpo_list = []
		if None in hpo_to_terms:
			none_hpo_list = hpo_to_terms[None]
			del hpo_to_terms[None]
		term_to_hpos = reverse_dict_list(hpo_to_terms)
		for non_hpo in none_hpo_list:
			if non_hpo in term_to_hpos:
				raise RuntimeWarning('Confilict: {}'.format(non_hpo))
			else:
				term_to_hpos[non_hpo] = []
		return term_to_hpos
		# hpo2depth = get_all_descendents_with_dist(ROOT_HPO_CODE, self.hpo_reader.get_slice_hpo_dict())
		# term_to_hpos = reverse_dict_list(hpo_to_terms)
		# for term, hpo_list in term_to_hpos.items():
		# 	_, hpo_list = zip(*sorted([(hpo2depth[hpo], hpo) for hpo in hpo_list]))  # 取最泛化的那个 or 全加入
		# 	term_to_hpos[term] = hpo_list
		# return term_to_hpos


	def sort_hpo_list(self, term, hpo_list):
		# FIXME:
		pass


	def get_hpo_to_std_terms(self):
		return SynGenerator().get_hpo_to_std_terms()


class ExactTermMatcher(TermMatcher):
	def __init__(self, hpo_to_terms, syn_dict_name, hpo_to_golds=None, hpo_to_gold_ps=None):
		"""
		Args:
			hpo_to_terms (dict): {hpo_code: term_list}
		"""
		super(ExactTermMatcher, self).__init__(hpo_to_golds, hpo_to_gold_ps)
		self.name = 'ExactTermMatcher-{}'.format(syn_dict_name)
		self.std_term_to_hpos = self.get_term_to_hpos(self.get_hpo_to_std_terms())
		self.term_to_hpos = self.get_term_to_hpos(hpo_to_terms)
		self.MAX_TERM_LENGTH = self.cal_max_term_length()


	def cal_max_term_length(self):
		len1 = max([len(term) for term in self.std_term_to_hpos]) if self.std_term_to_hpos else 0
		len2 = max([len(term) for term in self.term_to_hpos]) if self.term_to_hpos else 0
		return max(len1, len2)


	def get_max_term_length(self):
		return self.MAX_TERM_LENGTH


	def match(self, term):
		is_gold, gm, gs = self.match_gold(term)
		if is_gold:
			return gm, gs
		if term in self.std_term_to_hpos:
			return self.std_term_to_hpos[term], [1]*len(self.std_term_to_hpos[term])
		if term in self.term_to_hpos:
			return self.term_to_hpos[term], [1]*len(self.term_to_hpos[term])
		return None, 0


class BagTermMatcher(TermMatcher):
	def __init__(self, hpo_to_terms, syn_dict_name, hpo_to_golds=None, hpo_to_gold_ps=None):
		super(BagTermMatcher, self).__init__(hpo_to_golds, hpo_to_gold_ps)
		self.name = 'BagTermMatcher-{}'.format(syn_dict_name)
		self.std_analyzer = StandardAnalyzer()
		self.std_term_to_hpos = self.get_term_to_hpos(self.process_hpo_to_terms(self.get_hpo_to_std_terms()))
		self.term_to_hpos = self.get_term_to_hpos(self.process_hpo_to_terms(hpo_to_terms))
		self.MAX_TERM_LENGTH = self.cal_max_term_length()


	def cal_max_term_length(self):
		len1 = max([len(term) for term in self.std_term_to_hpos]) if self.std_term_to_hpos else 0
		len2 = max([len(term) for term in self.term_to_hpos]) if self.term_to_hpos else 0
		return max(len1, len2) + 2  # allow 2 empty char


	def get_max_term_length(self):
		return self.MAX_TERM_LENGTH


	def process_hpo_to_terms(self, hpo_to_terms):
		hpo_to_terms = deepcopy(hpo_to_terms)
		for hpo_code, term_list in hpo_to_terms.items():
			for i, term in enumerate(term_list):
				term_list[i] = self.get_bag_string(term)
		return hpo_to_terms


	def get_bag_string(self, term):
		return ''.join(sorted(self.std_analyzer.split(term)))


	def match(self, term):
		is_gold, gm, gs = self.match_gold(term)
		if is_gold:
			return gm, gs
		term = self.get_bag_string(term)
		if term in self.std_term_to_hpos:
			return self.std_term_to_hpos[term], [1]*len(self.std_term_to_hpos[term])
		if term in self.term_to_hpos:
			return self.term_to_hpos[term], [1]*len(self.term_to_hpos[term])
		return None, 0



if __name__ == '__main__':
	print(BagTermMatcher({}, '').std_term_to_hpos[''])








