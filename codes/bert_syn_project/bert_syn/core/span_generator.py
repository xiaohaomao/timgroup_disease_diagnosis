

import os
import json
from tqdm import tqdm
import itertools
from copy import deepcopy
import re
import numpy as np
import pandas as pd
from collections import Counter
#from multiprocessing import Pool
from billiard.pool import Pool # To avoid AssertionError: daemonic processes are not allowed to have children
import random
#import topwords


from bert_syn.utils.constant import DATA_PATH, RESULT_PATH

from bert_syn.core.baseline import DictSim
from bert_syn.utils.utils import get_file_list, get_all_descendents_for_many, contain_neg, contain_digits, dict_set_update, dict_list_add
from bert_syn.utils.utils import write_standard_file, read_standard_file, is_punc, is_space, contain_cns, timer, contain_digits
from bert_syn.utils.utils import reverse_dict, reverse_dict_list, cal_jaccard_sim_list
from bert_syn.core.data_helper import SynDictReader, SCELReader
from bert_syn.core.data_helper import HPOReader, get_del_words, get_stopwords, get_del_hpos, get_del_subwords


# ==========================================================================================
class StandardAnalyzer(object):
	def __init__(self):
		self.currentType = self.__getType('')


	def split(self, s):
		s += '.'
		splitList = []
		begin = 0
		for i in range(len(s)):
			type = self.__getType(s[i])
			if self.currentType == 0:
				begin = i
			elif type > 2 or type != self.currentType:
				splitList.append(s[begin:i])
				begin = i
			self.currentType = type
		return splitList


	def __getType(self, c):
		if self.__isNum(c):			#
			return 1
		elif self.__isENG(c):		#
			return 2
		elif self.__isCNS(c):		#
			return 3
		else:
			return 0

	def __isNum(self, c):
		return '\u0030' <= c <='\u0039'


	def __isENG(self, c):
		return ('\u0041' <= c <='\u005a') or ('\u0061' <= c <='\u007a')


	def __isCNS(self, c):
		return '\u4e00' <= c <= '\u9fff'


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


# ==========================================================================================
class NaiveExactTermMatcher(object):
	def __init__(self, vocab):
		self.name = 'NaiveExactTermMatcher'
		self.vocab = set(vocab)
		self.MAX_TERM_LENGTH = self.cal_max_term_length()


	def cal_max_term_length(self):
		return max([len(term) for term in self.vocab]) if self.vocab else 0


	def get_max_term_length(self):
		return self.MAX_TERM_LENGTH


	def match(self, term):
		"""
		Args:
			term (str)
		Returns:
			list: [info, ...]
			list: [score, ...]
		"""
		if term in self.vocab:
			return [term], [1]
		return None, None


# ==========================================================================================
class NaiveBagTermMatcher(object):
	def __init__(self, vocab):
		self.name = 'NaiveBagTermMatcher'
		self.analyzer = StandardAnalyzer()
		self.vocab = self.process_vocab(vocab)
		self.MAX_TERM_LENGTH = self.cal_max_term_length()


	def process_vocab(self, vocab):
		return set([self.get_bag_string(t) for t in vocab])


	def get_bag_string(self, s):
		return ''.join(sorted(self.analyzer.split(s)))


	def cal_max_term_length(self):
		return max([len(term) for term in self.vocab]) if self.vocab else 0


	def get_max_term_length(self):
		return self.MAX_TERM_LENGTH


	def match(self, term):
		term = self.get_bag_string(term)
		if term in self.vocab:
			return [term], [1]
		return None, None


# ==========================================================================================
class ExactTermMatcher(object):
	def __init__(self, word2codes=None, include_paths=None, exclude_paths=None):
		"""
		Args:
			word2codes (dict): {word: set([hpo1, hpo2, ...]}
		"""
		super(ExactTermMatcher, self).__init__()
		self.name = 'ExactTermMatcher'
		self._word2codes = {}
		if word2codes is not None:
			self.add_vocab_from_word2codes(word2codes)
		include_paths = include_paths or []
		exclude_paths = exclude_paths or []
		for path in include_paths:
			self.add_vocab_from_path(path)
		for path in exclude_paths:
			self.remove_vocab_from_path(path)
		self.MAX_TERM_LENGTH = self.cal_max_term_length()


	def get_word_key(self, term):
		return term


	def add_vocab_from_word2codes(self, word2codes):
		for word, codes in word2codes.items():
			dict_set_update(self.get_word_key(word), codes, self._word2codes)


	def add_vocab_from_path(self, path):
		"""
		Args:
			path (str): 词典文件路径，每行格式：code | term
		"""
		infos = read_standard_file(path)
		for code, word in infos:
			dict_list_add(self.get_word_key(word), code, self._word2codes)


	def remove_vocab_from_path(self, path):
		infos = read_standard_file(path)
		for code, word in infos:
			word = self.get_word_key(word)
			if word in self._word2codes and code in self._word2codes[word]:
				self._word2codes[word].remove(code)
				if len(self._word2codes[word]) == 0:
					del self._word2codes[word]


	def cal_max_term_length(self):
		return max([len(term) for term in self._word2codes]) if self._word2codes else 0


	def get_max_term_length(self):
		return self.MAX_TERM_LENGTH


	def match(self, term):
		"""
		Args:
			term (str)
		Returns:
			list: [hpo_code, ...]
			list: [score, ...]
		"""
		term = self.get_word_key(term)
		if term in self._word2codes:
			return list(self._word2codes[term]), [1]*len(self._word2codes[term])
		return None, None


class BagTermMatcher(ExactTermMatcher):
	def __init__(self, word2codes=None, include_paths=None, exclude_paths=None):
		self.name = 'BagTermMatcher'
		self.analyzer = StandardAnalyzer()
		super(BagTermMatcher, self).__init__(word2codes, include_paths, exclude_paths)


	def get_word_key(self, s):
		return ''.join(sorted(self.analyzer.split(s)))


# ==========================================================================================
def score_ary_to_result(score_ary, col_names, score_thres, match_type, word2codes):
	"""
	Args:
		score_ary:
		col_names:
	Returns:
		list: [code1, ...]
		list: [score, ...]
	"""
	if score_thres is None:
		term_score_pairs = list(zip(col_names, score_ary))
	else:

		term_score_pairs = [(tgt_term, score) for tgt_term, score in zip(col_names, score_ary) if score > score_thres]
	if len(term_score_pairs) == 0:
		return [], []
	if match_type == 'all':
		term_score_pairs = sorted(term_score_pairs, key=lambda item: item[1], reverse=True)
	else:
		term_score_pairs = [max(term_score_pairs, key=lambda item: item[1])]
	ret_codes, ret_scores = [], []
	for term, score in term_score_pairs:
		codes = word2codes[term]
		ret_codes.extend(codes)
		ret_scores.extend([score]*len(codes))
	return ret_codes, ret_scores


def score_ary_to_result_wrapper(paras):
	return score_ary_to_result(*paras)


class ScoreThresholdMatcher(object):
	def __init__(self, word2codes, score_thres=None, match_type='all', cpu_use=12, stopwords=None, verbose=False):
		"""
		Args:
			word2codes (dict): {word: [code1, code2, ...], ...}
			match_type (str): 'all' | 'best'
		"""
		self.word2codes = word2codes
		self.tgt_words = list(self.word2codes.keys())
		self.score_thres = score_thres or -np.inf
		self.match_type = match_type
		self.cpu_use = cpu_use
		if stopwords:
			self.stopwords_pattern = re.compile('|'.join(stopwords))
		self.verbose = verbose
		self.hpo2text = HPOReader().get_hpo_to_cns()



	def set_score_threshold(self, score_thres):
		self.score_thres = score_thres


	def predict_scores(self, terms):
		"""
		Returns:
			np.ndarray: shape=(len(terms), len(tgt_words))
			list: column_names;
		"""
		raise NotImplementedError


	def process_term(self, term):

		return term.strip()


	def match(self, term):
		"""
		Args:
			term (str)
		Returns:
			list: [code1, ...]
			list: [score, ...]
		"""
		score_mat, col_names = self.predict_scores([term])
		score_ary = score_mat[0]
		return score_ary_to_result(score_ary, col_names, self.score_thres, self.match_type, self.word2codes)


	@timer
	def match_many(self, terms):
		"""
		Args:
			terms (list): [str1, str2, ...]
		Returns:
			list: [result1, result2, ...]; result = ([code, ...], [score, ...])
		"""
		def get_iterator(score_mat, col_names):
			for i in tqdm(range(score_mat.shape[0])):
				yield score_mat[i], col_names, self.score_thres, self.match_type, self.word2codes
		if len(terms) == 0:
			return []
		terms = [self.process_term(t) for t in terms]
		score_mat, col_names = self.predict_scores(terms)
		if self.cpu_use == 1:
			results = [score_ary_to_result_wrapper(paras) for paras in get_iterator(score_mat, col_names)]
		else:
			with Pool(self.cpu_use) as pool:
				results = pool.map(score_ary_to_result_wrapper, get_iterator(score_mat, col_names), chunksize=100)
		if self.verbose:
			self.print_samples(terms, results, 300)
		return results


	def print_samples(self, terms, results, sample_num):
		sample_ranks = random.sample(range(len(terms)), min(len(terms), sample_num))
		for idx in sample_ranks:
			hpo_codes, scores = results[idx]
			print('{} -> {} ({})'.format(terms[idx], [self.hpo2text.get(hpo, '') for hpo in hpo_codes], scores))


class AlbertDDMLMatcher(ScoreThresholdMatcher):
	def __init__(self, word2codes, model, score_thres=None, match_type='all', cpu_use=12, stopwords=None, verbose=False):
		"""
		Args:
			model (BertDDMLSim)
		"""
		super(AlbertDDMLMatcher, self).__init__(word2codes, score_thres, match_type, cpu_use, stopwords, verbose)
		self.model = model
		self.model.set_dict_terms(self.tgt_words)
		self.model.get_dict_embedding()


	def predict_scores(self, terms):
		"""
		Returns:
			np.ndarray: shape=(len(terms), len(tgt_words))
			list: column_names;
		"""
		return self.model.predict_scores(terms, cpu_use=self.cpu_use)


class JaccardMatcher(ScoreThresholdMatcher):
	def __init__(self, word2codes, score_thres=None, match_type='all', cpu_use=12, stopwords=None, jaccard_sym=True, verbose=False):
		super(JaccardMatcher, self).__init__(word2codes, score_thres, match_type, cpu_use, stopwords, verbose)
		self.jaccard_sym = jaccard_sym


	def predict_scores(self, terms):
		"""
		Returns:
			np.ndarray: shape=(len(terms), len(tgt_words))
			list: column_names;
		"""
		pairs = []
		for src_term in terms:
			pairs.extend([(src_term, tgt_term) for tgt_term in self.tgt_words])
		scores = cal_jaccard_sim_list(pairs, sym=self.jaccard_sym, cpu_use=self.cpu_use)
		assert len(scores) == len(pairs)
		step = len(self.tgt_words)
		ret_ary = []
		for i in range(0, len(scores), step):
			ret_ary.append(scores[i: i+step])
		return np.vstack(ret_ary), self.tgt_words


# ==========================================================================================
class TextSearcher(object):
	def __init__(self):
		self.SPLIT_SEN_PATTEN = re.compile(r'[，。！？；\n\r\t]')
		self.neg_terms = [
			'未引出', '未见', '未再', '未出现', '未诉', '未见异常', '未闻及', '未及', '未闻', '未查', '未累及', '未予', '未显示', '未',
			'无', '无明显', '不明显', '非', '没有',
			'否认',  '阴性', '不符合', '不考虑', '除外', '不伴',
			'（-）', '(-)', '（－）', '(－)',
		]
		self.not_neg_terms = [
			'无明显诱因'
		]
		self.stop_words = {'有时', '的', '较前', '稍'}


	def search_sentence(self, text):
		"""
		Args:
			text (string)
		Returns:
			list: hpo_list; [hpo_code1, hpo_code2, ...]
			list: pos_list; [np.array([begin1, end1]), ...]
		"""
		raise NotImplementedError


	def set_neg_terms(self, neg_terms):
		self.neg_terms = neg_terms


	def set_split_sen_pattern(self, split_pattern):
		self.SPLIT_SEN_PATTEN = re.compile(split_pattern)


	def set_stop_words(self, stop_words):
		self.stop_words = set(stop_words)


	def add_stop_words(self, stop_words):
		self.stop_words.update(stop_words)


	def search(self, doc):
		"""
		Args:
			doc (str)
		Returns:
			list: hpo_list; [code1, code2, ...]
			list: pos_list; [np.array([begin1, end1]), ...]
		"""
		ret_hpo_list, ret_pos_list = [], []
		sen_list, offset_list = self.doc_to_sen_list(doc, self.SPLIT_SEN_PATTEN, split_len=1)
		for i in range(len(sen_list)):
			hpo_list, pos_list = self.search_sentence(sen_list[i])
			hpo_list, pos_list = self.neg_detect_sen_filter(hpo_list, pos_list, sen_list[i])
			for k in range(len(pos_list)):
				pos_list[k] += offset_list[i]
			ret_hpo_list.extend(hpo_list)
			ret_pos_list.extend(pos_list)
		return self.sort_result(ret_hpo_list, ret_pos_list)


	def neg_detect_sen_filter(self, match_hpos, pos_list, sentence):
		"""Negative detection
		"""
		match_hpos, pos_list = self.sort_result(match_hpos, pos_list)
		mpos_list = merge_pos_list(pos_list)
		rest_str, p = '', 0
		for b, e in mpos_list:
			rest_str += sentence[p: b]
			p = e
		rest_str += sentence[p: len(sentence)]
		for not_neg_term in self.not_neg_terms:
			if rest_str.find(not_neg_term) > -1:
				return match_hpos, pos_list
		for neg_term in self.neg_terms:
			if rest_str.find(neg_term) > -1:
				return [], []
		return match_hpos, pos_list


	def sort_result(self, match_hpos, pos_list):
		if len(match_hpos) == 0:
			return match_hpos, pos_list
		match_hpos, pos_list = zip(*sorted(zip(match_hpos, pos_list), key=lambda item: item[1][0]))
		return list(match_hpos), list(pos_list)


	def doc_to_sen_list(self, doc, pattern, split_len=1):
		"""
		Returns:
			list: list of sentences
			list: [sen1_begin, sen2_begin, ...], offset of sentence
		"""
		sen_list = pattern.split(doc)
		offset_list = [0]
		for sen in sen_list:
			offset_list.append(offset_list[-1]+len(sen)+split_len)
		offset_list.pop()
		return sen_list, offset_list


# ==========================================================================================
class MaxInvTextSearcher(TextSearcher):
	def __init__(self, term_matcher, invalid_end_set=None, skip_terms=None):
		super(MaxInvTextSearcher, self).__init__()
		self.matcher = term_matcher
		self.max_term_length = term_matcher.get_max_term_length()
		self.name = '{}-{}'.format('MaxInvTextSearcher', self.matcher.name)

		self.invalid_end_set = invalid_end_set or {}
		self.skip_terms = skip_terms or {}


	def _search_sentence(self, text, matcher, max_len, skip_terms, invalid_begin_end):
		"""
				Args:
					text (string)
				Returns:
					list: hpo_list; [code1, code2, ...]
					list: pos_list; [np.array([begin1, end1]), ...]
				"""
		info_list, pos_list = [], []
		end = len(text)
		while end != 0:
			for begin in range(max(0, end - max_len), end):
				term = text[begin: end]
				if term in skip_terms:
					end = begin + 1
					break
				if invalid_begin_end(term):
					continue
				match_list, _ = matcher.match(term)
				if match_list:
					info_list.extend(match_list)
					pos_list.extend([np.array([begin, end]) for _ in match_list])
					end = begin + 1
					break
			end -= 1
		return info_list, pos_list


	def search_sentence(self, text):
		return self._search_sentence(text, self.matcher, self.max_term_length, self.skip_terms, self.invalid_begin_end)


	def invalid_begin_end(self, term):
		return len(term) == 0 or self.invalid_begin(term[0]) or self.invalid_end(term[-1])


	def invalid_begin(self, c):
		return is_punc(c) or is_space(c)


	def invalid_end(self, c):
		if c in self.invalid_end_set:
			return False
		return is_punc(c) or is_space(c)


class DictAndRankerTextSearcher(MaxInvTextSearcher):
	def __init__(self, dict_term_matcher):
		super(DictAndRankerTextSearcher, self).__init__(dict_term_matcher)
		self.del_words = get_del_words()
		del_subwords = list(get_del_subwords())
		self.del_subwords_pattern = re.compile('|'.join(del_subwords))
		self.sub_sent_split_pattern = re.compile('、')


	def qualified_cand_sent(self, sent):
		for neg_term in self.neg_terms:
			if sent.find(neg_term) > -1:
				return False
		if len(sent) < 2:
			return False
		if not contain_cns(sent):
			return False
		if re.search('mol|mg|g/|mm/H|[mM][lL]|/h|/u[lL]|/L|mmHg|↑|↓|次/分', sent):
			return False
		sent = sent.strip()
		if sent in self.del_words:
			return False
		if self.del_subwords_pattern.search(sent):
			return False
		return True


	def search_ret_cands(self, doc):
		min_win = 12

		ret_hpo_list, ret_pos_list = [], []
		sen_list, offset_list = self.doc_to_sen_list(doc, self.SPLIT_SEN_PATTEN, split_len=1)
		cand_sents, cand_offsets, rank_interval = [], [], []
		for i in range(len(sen_list)):
			hpo_list, pos_list = self.search_sentence(sen_list[i])
			filtered_hpo_list, filtered_pos_list = self.neg_detect_sen_filter(hpo_list, pos_list, sen_list[i])
			contain_neg = len(hpo_list) > len(filtered_hpo_list)
			if not contain_neg and len(filtered_hpo_list) == 0 and self.qualified_cand_sent(sen_list[i]):
				b = len(cand_sents)
				cand_sents.append(sen_list[i])
				cand_offsets.append(offset_list[i])


				e = len(cand_sents)
				rank_interval.append((b, e))
			for k in range(len(filtered_pos_list)):
				filtered_pos_list[k] += offset_list[i]
			ret_hpo_list.extend(filtered_hpo_list)
			ret_pos_list.extend(filtered_pos_list)
		return ret_hpo_list, ret_pos_list, cand_sents, cand_offsets, rank_interval



	def search_with_second_matcher(self, doc, second_matcher):
		"""
		Args:
			doc (str)
		Returns:
			list: hpo_list; [code1, code2, ...]
			list: pos_list; [np.array([begin1, end1]), ...]
		"""
		return self.search_multi_with_second_matcher([doc], second_matcher)[0]


	def extract_entity_from_list(self,doc):
		pos_span,text_list = [],[]

		for i_term in range(len(doc)):

			pos_span.extend(doc[i_term]['SPAN_LIST'])
			text_list.append(doc[i_term]['SPAN_TEXT'])

		return text_list,pos_span,len(doc)


	def search_multi_with_second_matcher(self, docs, second_matcher, entity_method, cpu_use=12):
		"""
		Args:
			docs (list): [str, str, ...]
		Returns:
			list: [(hpo_list, pos_list), ...]; length = len(docs)
		"""
		# 这里是将所有的text 排列成一个list，做了一些否定检测
		doc_sent_nums, sent_list,doc_entity_num = [], [],[]
		if entity_method == 'unsupervised':

			if cpu_use == 1:
				results = [self.search_ret_cands(doc) for doc in docs]
			else:
				with Pool(cpu_use) as pool:
					results = pool.map(self.search_ret_cands, docs)

			for doc_id, (_, _, cand_sents, _, _) in enumerate(results):
				doc_sent_nums.append(len(cand_sents))
				sent_list.extend(cand_sents)


			sent_match_results = second_matcher.match_many(sent_list)

			sent_id = 0
			ret_results = []
			for doc_id in range(len(doc_sent_nums)):
				doc_hpo_list, doc_pos_list, doc_cand_sents, doc_cand_offsets, doc_rank_interval = results[doc_id]
				for b, e in doc_rank_interval:
					best_sent, best_offset, best_codes, best_scores, cmp_score = None, None, None, None, -np.inf
					for i in range(b, e):
						sent, offset = doc_cand_sents[i], doc_cand_offsets[i]
						codes, scores = sent_match_results[sent_id];
						sent_id += 1
						if len(codes) > 0 and scores[0] > cmp_score:
							cmp_score = scores[0]
							best_sent, best_offset, best_codes, best_scores = sent, offset, codes, scores
					if best_sent is not None:
						doc_hpo_list.extend(best_codes)
						doc_pos_list.extend([np.array([best_offset, best_offset + len(best_sent)]) for _ in best_codes])
				ret_results.append(self.sort_result(doc_hpo_list, doc_pos_list))
			return ret_results

		elif entity_method == 'NERorTAG':

			results = [self.extract_entity_from_list(doc) for doc in docs]



			for doc_id, (text_list,span_list,doc_length) in enumerate(results):
				doc_sent_nums.append(span_list)
				sent_list.extend(text_list)
				doc_entity_num.append(doc_length)

			sent_match_results = second_matcher.match_many(sent_list)




			ret_results = []
			ret_span_num = 0
			for doc_num in range(len(results)):

				if results[doc_num][-1]==0:
					ret_results.append(([], []))
				else:
					hpo_code_list , hpo_span_list = [], [ ]
					for i_entity in range(ret_span_num,ret_span_num+results[doc_num][-1]):
						print('i_entity',i_entity)
						if sent_match_results[i_entity][0]!=[] or len(sent_match_results[i_entity][0])>0:
							hpo_code_list.append(sent_match_results[i_entity][0][0])
							hpo_span_list.append(np.array(results[doc_num][1][i_entity-ret_span_num]))
							#hpo_span_list = np.append(hpo_span_list,results[doc_num][1][i_entity])
					ret_results.append((hpo_code_list,hpo_span_list))
				ret_span_num+= results[doc_num][-1]
			return ret_results
		else:
			assert False








# ==========================================================================================
class SentSearcherOld(TextSearcher):
	def __init__(self, term_matcher, vocab_split_pattern=None, vocab_filter_func=None):
		super(SentSearcherOld, self).__init__()
		self.term_matcher = term_matcher
		assert hasattr(term_matcher, 'match_many')
		self.vocab_split_pattern = vocab_split_pattern or re.compile(r'[、，。！？；\r\n\t,!?;]')
		self.vocab_filter_func = vocab_filter_func or self.default_vocab_filter


	def default_vocab_filter(self, term):
		sent = term.strip()
		# sent = re.sub('\d+[年月日天]|(\d{4}|\d{1,2})-\d{1,2}(-\d{1,2})?', '', sent)
		if len(sent) < 2:
			return False
		if re.search('mol|mg|g/|mm/H|[mM][lL]|/h|/u[lL]|/L|mmHg|↑|↓|次/分', sent):
			return False
		if not contain_cns(sent):
			return False
		if contain_digits(sent):
			return False
		return True


	def get_vocab_list(self, doc):
		"""
		Args:
			doc (str)
		Returns:
			list: [[str1, str2,...], ...]
		"""
		doc = re.sub('\d+[年月日天号]|(\d{4}|\d{1,2})-\d{1,2}(-\d{1,2})?', '', doc)   # remove time stamp
		short_sents = self.vocab_split_pattern.split(doc)
		short_sents = [s.strip() for s in short_sents if self.vocab_filter_func(s)]
		min_term_len = 2
		vocab_list = []
		for s in short_sents:
			vocab = set()
			for i in range(len(s)-min_term_len+1):
				for j in range(i+min_term_len, len(s)+1):
					vocab.add(s[i: j].strip())
			vocab = [term for term in vocab if self.vocab_filter_func(term)]
			vocab_list.append(vocab)
		return vocab_list


	@timer
	def get_word2codes(self, doc, save_txt=None):
		"""
		Args:
			doc:
		Returns:
			dict: {term: set([code1, code2, ...])}
		"""
		vocab_list = self.get_vocab_list(doc)
		terms, be_list = [], []
		for vocab in vocab_list:
			be_list.append((len(terms), len(terms)+len(vocab)))
			terms.extend(vocab)

		if len(terms) == 0:
			return {}
		results = self.term_matcher.match_many(terms)
		word2codes = {}
		selected_terms, selected_codes, selected_scores = [], [], []
		for b, e in be_list:
			sub_results = results[b:e]
			sub_terms = terms[b:e]
			score_max = -np.inf
			best_term, best_codes, best_scores = None, None, None
			for term, (code_list, score_list) in zip(sub_terms, sub_results):
				if not code_list:
					continue
				best_idx = np.argmax(score_list)
				if score_list[best_idx] > score_max:
					score_max = score_list[best_idx]
					best_term, best_codes, best_scores = term, code_list, score_list
			if best_term is not None:
				word2codes[best_term] = best_codes
				selected_terms.append(best_term); selected_codes.append(best_codes); selected_scores.append(best_scores)

		if save_txt is not None:
			def sort_func(item):
				if not item[2]:
					return -np.inf
				return item[2][0]
			os.makedirs(os.path.dirname(save_txt), exist_ok=True)
			hpo_to_cns = HPOReader().get_hpo_to_cns()
			item_lists = []
			for term, code_list, score_list in zip(selected_terms, selected_codes, selected_scores):
				if code_list:
					item_lists.append((term, [f'{code}-{hpo_to_cns.get(code, "")}' for code in code_list], score_list))
			item_lists = sorted(item_lists, key=sort_func, reverse=True)
			write_standard_file(item_lists, save_txt, split_char=' | ')
		return word2codes





	def search(self, doc, save_word2codes_txt=None):
		"""
		Args:
			doc (str)
		Returns:
			list: code_list; [code1, code2, ...]
			list: pos_list; [np.array([begin1, end1]), ...]
		"""
		word2codes = self.get_word2codes(doc, save_txt=save_word2codes_txt)
		if len(word2codes) == 0:
			return [], []
		exact_matcher = ExactTermMatcher(word2codes=word2codes)
		max_inv_searcher = MaxInvTextSearcher(exact_matcher)
		return max_inv_searcher.search(doc)


	def search_multi(self, docs, save_word2codes_txt=None, cpu_use=12):
		"""
		Args:
			docs (list)
		Returns:
			list: [search_result1, ...]; search_result = (code_list, pos_list)
		"""
		word2codes = self.get_word2codes('\n'.join(docs), save_txt=save_word2codes_txt)
		if len(word2codes) == 0:
			return [], []
		exact_matcher = ExactTermMatcher(word2codes=word2codes)
		max_inv_searcher = MaxInvTextSearcher(exact_matcher)
		with Pool(cpu_use) as pool:
			return pool.map(max_inv_searcher.search, docs)


# ==========================================================================================
class CorpusGenerator(object):
	def __init__(self):
		pass


# ==========================================================================================
class PUMC2000CorpusGenerator(CorpusGenerator):
	def __init__(self, name=None, sent_split_pattern=None, sent_filter_func=None):
		super(PUMC2000CorpusGenerator, self).__init__()
		self.name = name or 'corpus'
		self.RAW_XLSX = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20-RareDisease-master/pumc_tag/project/data/raw/罕见病数据-总.xlsx'
		self.remove_patient_folder = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/pumc_tag/project/data/tag_json/case87-doc-hy-strict-enhance'
		self.SAVE_TXT = os.path.join(DATA_PATH, 'preprocess', 'pumc_2000', f'{self.name}.txt')
		os.makedirs(os.path.dirname(self.SAVE_TXT), exist_ok=True)
		self.sent_split_pattern = sent_split_pattern or re.compile(r'[、，：。！？；”“（）【】〔〕{}\r\n\t,!:?;"()+\[\]]')
		self.sent_filter_func = sent_filter_func or self.default_sent_filter


	def get_remove_patient_info(self, input_folder):
		"""
		Args:
			input_folder (str)
		Returns:
			set: {(病案号, 姓名), ...}
		"""
		json_list = get_file_list(input_folder, lambda p:p.endswith('.json'))
		info_set = set()
		for json_path in json_list:
			field_info = json.load(open(json_path))
			info_set.add((field_info['病案号']['RAW_TEXT'].strip(), field_info['姓名']['RAW_TEXT'].strip()))
		return info_set


	def get_corpus(self, remove_case87=False):
		remove_info_set = None
		if remove_case87:
			remove_info_set = self.get_remove_patient_info(self.remove_patient_folder)
		df = pd.read_excel(self.RAW_XLSX, '入院患者数据')
		patients = df.to_dict(orient='records')  # [{'登记号': str, '现病史': str, ...}, ...]
		input_field = ['现病史', '既往史', '家族史', '入院情况', '出院情况', '出院诊断']
		sents = []
		for pa_info in tqdm(patients):
			if remove_info_set is not None and (pa_info['病案号'].strip(), pa_info['姓名'].strip()) in remove_info_set:
				continue
			for field in input_field:
				s = pa_info[field]
				if not isinstance(s, str): continue
				sents.extend(self.split_sents(s))
		return sents


	def gen_corpus(self, remove_case87=False, del_repeat=False):
		lines = self.get_corpus(remove_case87)
		if del_repeat:
			lines = list(set(lines))
		print('Total lines: {}'.format(len(lines)))
		print('Saved in {}'.format(self.SAVE_TXT))
		open(self.SAVE_TXT, 'w').write('\n'.join(lines))


	def default_sent_filter(self, sent):
		sent = sent.strip()
		# sent = re.sub('\d+[年月日天]|(\d{4}|\d{1,2})-\d{1,2}(-\d{1,2})?', '', sent)
		if len(sent) < 2:
			return False
		if re.search('mol|mg|g/|mm/H|[mM][lL]|/h|/u[lL]|/L|mmHg|/[Kk]g|↑|↓|次/分', sent):
			return False
		if not contain_cns(sent):
			return False
		return True


	def split_sents(self, s):
		"""
		Args:
			s (str):
		Returns:
			list: [sentence1, ...]
		"""
		if self.sent_split_pattern is None:
			return s
		sents = self.sent_split_pattern.split(s)
		sents = [s.strip() for s in sents]
		ret_sents = [sent for sent in sents if self.sent_filter_func(sent)]
		return ret_sents


# ==========================================================================================
class VocabGenerator(object):
	def __init__(self, name=None):
		self.name = name or 'naive_count'
		self.SAVE_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'pumc_2000')
		os.makedirs(os.path.dirname(self.SAVE_FOLDER), exist_ok=True)


	def get_save_txt(self, mark=None):
		file_name = f'{self.name}-{mark}-vocab.txt' if mark is not None else f'{self.name}-vocab.txt'
		return os.path.join(self.SAVE_FOLDER, file_name)


	def read_corpus(self, path):
		"""
		Args:
			path (str): one sentence each line
		Returns:
			list: [sentence1, sentence2, ...]
		"""
		sents = open(path).read().splitlines()
		return [sent.strip() for sent in sents]


class NaiveCountVocabGenerator(VocabGenerator):
	def __init__(self, name=None):
		super(NaiveCountVocabGenerator, self).__init__(name or 'naive_count')


	def gen_vocab(self, corpus, min_count=2, min_len=2, mark=None):
		"""
		Args:
			corpus (str or list):
				str: path
				list: [sentence1, sentence2, ...]
		"""
		if isinstance(corpus, str):
			corpus = self.read_corpus(corpus)
		counter = Counter(corpus)
		term_count_list = []
		for term, count in counter.most_common():
			if count < min_count:
				continue
			if len(term) < min_len:
				continue
			if contain_digits(term):
				continue
			if contain_neg(term):
				continue
			if not contain_cns(term):
				continue
			term_count_list.append((term, count))
		term_score_pairs = sorted(term_count_list, key=lambda item:item[1], reverse=True)
		open(self.get_save_txt(mark=mark), 'w').write('\n'.join([term for term, _ in term_score_pairs]))
		write_standard_file(term_score_pairs, self.get_save_txt('freq' if mark is None else f'freq-{mark}'), split_char=' | ')


class TopwordsVocabGenerator(VocabGenerator):
	def __init__(self, name=None):
		super(TopwordsVocabGenerator, self).__init__(name or 'topwords')


	def filter_vocab(self, terms):
		"""
		Args:
			terms (iterable): [term1, term2, ...]
		Returns:
			set: {term1, term2, ...}
		"""
		ret_term_set = set()
		for term in terms:
			if len(term) < 2:
				continue
			if not contain_cns(term):
				continue
			if contain_digits(term):
				continue
			if contain_neg(term):
				continue
			ret_term_set.add(term)
		return ret_term_set


	def gen_vocab(self, corpus, max_len=8, n_iter=5, lamb=1e-5, freq_threshold=1e-3, n_jobs=12, verbose=True, mark=None):
		"""
		Args:
			corpus (str or list):
				str: path
				list: [sentence1, sentence2, ...]
		"""
		if isinstance(corpus, str):
			corpus = self.read_corpus(corpus)
		vocab2freq, vocab2psi = topwords.em(corpus, n_iter=n_iter, freq_threshold=freq_threshold,
			max_len=max_len, lamb=lamb, verbose=verbose, n_jobs=n_jobs)
		keep_terms = self.filter_vocab(list(vocab2freq.keys()))
		term_score_pairs = sorted(
			[(term, score) for term, score in vocab2freq.items() if term in keep_terms],
			key=lambda item: item[1], reverse=True)
		open(self.get_save_txt(mark=mark), 'w').write('\n'.join([term for term, _ in term_score_pairs]))
		write_standard_file(term_score_pairs, self.get_save_txt('freq' if mark is None else f'freq-{mark}'), split_char=' | ')


# ==========================================================================================
class SpanGenerator(object):
	def __init__(self, vocab, match_type='exact', search_type='max_inv'):
		"""
		Args:
			vocab (list of set of terms):
			match_type (str): 'exact' | 'bag'
			search_type (str): 'max_inv'
		"""
		if match_type == 'exact':
			matcher = NaiveExactTermMatcher(vocab)
		elif match_type == 'bag':
			matcher = NaiveBagTermMatcher(vocab)
		else:
			raise RuntimeError('Unknown match type: {}'.format(match_type))

		if search_type == 'max_inv':
			self.searcher = MaxInvTextSearcher(matcher)
		else:
			raise RuntimeError('Unknown search type: {}'.format(search_type))


	def process(self, doc):
		"""
		Args:
			doc (str)
		Returns:
			list: [
					{
						'SPAN_LIST': [(start_pos, end_pos), ...],
						'SPAN_TEXT': str
					},
					...
				]
		"""
		info_list, span_list = self.searcher.search(doc)
		ret = []
		for span in span_list:
			b, e = int(span[0]), int(span[1])
			ret.append({'SPAN_LIST': [(b, e)], 'SPAN_TEXT': doc[b: e]})
		return ret


# ==========================================================================================
def search_result_to_entity_list(text, search_result, hpo2text):
	"""
	Returns:
		list: [
			{
				'SPAN_LIST': [(start_pos, end_pos), ...],
				'SPAN_TEXT': str
				'HPO_CODE': str,
				'HPO_TEXT': str,
				'TAG_TYPE': str
			}
		]
	"""
	hpo_list, span_ary_list = search_result

	assert len(hpo_list) == len(span_ary_list)
	#quit()
	ret_list = []
	for hpo, span_ary in zip(hpo_list, span_ary_list):



		ret_list.append({
			'SPAN_LIST':[span_ary.tolist()],
			'SPAN_TEXT':text[int(span_ary[0]): int(span_ary[-1])],

			'HPO_CODE':hpo,
			'HPO_TEXT':hpo2text[hpo],
		})
	return ret_list


def combine_word2hpos(*word2hpos_list):
	word2hpos = deepcopy(word2hpos_list[0])
	for word in word2hpos:
		word2hpos[word] = set(word2hpos[word])
	for d in word2hpos_list[1:]:
		for word, hpos in d.items():
			dict_set_update(word, hpos, word2hpos)
	return word2hpos


def process_word2hpos(word2hpos):
	del_hpos = get_del_hpos()
	del_words = get_del_words()

	ret_word2hpos = {}
	for w, hpos in word2hpos.items():
		w = w.strip()
		if len(w) < 2:
			continue
		if w in del_words:
			print('remove:', w)
			continue
		hpos = set(hpo for hpo in hpos if hpo not in del_hpos)
		ret_word2hpos[w] = hpos
	return ret_word2hpos


def run_sent_match(input_folder, output_folder, model_name, global_step=None,
		score_thres=None, match_type='all', cpu_use=12, multi=True, save_word2codes_txt=None, fchunk=10):
	"""Note:
		field_to_info: {
			FIELD: {
				'RAW_TEXT': str,
				'ENTITY_LIST': [
					{
						'SPAN_LIST': [(start_pos, end_pos), ...],
						'SPAN_TEXT': str
						'HPO_CODE': str,
						'HPO_TEXT': str,
						'TAG_TYPE': str
					},
					...
				]
			}
		}
	"""
	from bert_syn.core.bert_ddml_sim import BertDDMLSim
	bert_sim = BertDDMLSim(model_name)
	bert_sim.restore(global_step=global_step)

	hpo_reader = HPOReader()
	hpo2text = hpo_reader.get_hpo_to_cns()
	matcher = AlbertDDMLMatcher(process_word2hpos(hpo_reader.get_cns_to_hpo()), bert_sim,
		score_thres=score_thres, match_type=match_type, cpu_use=cpu_use, stopwords=get_stopwords())
	searcher = SentSearcherOld(matcher)


	all_input_jsons = sorted(get_file_list(input_folder, lambda p: p.endswith('.json')))
	if multi:
		assert hasattr(searcher, 'search_multi')
		for input_idx in range(0, len(all_input_jsons), fchunk):
			input_jsons = all_input_jsons[input_idx: input_idx+fchunk]
			key_to_text = {}    # {(input_json, field_name): raw_text}
			for input_json in input_jsons:
				field_to_info = json.load(open(input_json))
				for field, info in field_to_info.items():
					key_to_text[(input_json, field)] = info['RAW_TEXT']
			keys, texts = zip(*key_to_text.items())
			txt = os.path.splitext(save_word2codes_txt)[0]+f'-{input_idx}_{input_idx+fchunk}.txt'
			search_results = searcher.search_multi(texts, save_word2codes_txt=txt, cpu_use=cpu_use)
			outjson_to_dict ={}
			for (input_json, field), text, search_result in tqdm(zip(keys, texts, search_results)):
				entity_list = search_result_to_entity_list(text, search_result, hpo2text)
				output_json = input_json.replace(input_folder, output_folder)
				if output_json not in outjson_to_dict:
					outjson_to_dict[output_json] = {}
				outjson_to_dict[output_json][field] = {'RAW_TEXT': text, 'ENTITY_LIST': entity_list}
			for outjson, field_to_info in outjson_to_dict.items():
				os.makedirs(os.path.dirname(outjson), exist_ok=True)
				json.dump(field_to_info, open(outjson, 'w'), indent=2, ensure_ascii=False)
			print('processed json file: {}/{}'.format(input_idx + fchunk, len(all_input_jsons)))
	else:
		for input_json in all_input_jsons:
			output_json = input_json.replace(input_folder, output_folder)
			os.makedirs(os.path.dirname(output_json), exist_ok=True)
			field_to_info = json.load(open(input_json))
			for field, info in field_to_info.items():
				text = info['RAW_TEXT']
				search_result = searcher.search(text)
				entity_list = search_result_to_entity_list(text, search_result, hpo2text)
				info['ENTITY_LIST'] = entity_list
			json.dump(field_to_info, open(output_json, 'w'), indent=2, ensure_ascii=False)


def gen_best_match(src_terms, save_txt, model_name, global_step=None):
	from bert_syn.core.bert_ddml_sim import BertDDMLSim
	bert_sim = BertDDMLSim(model_name)
	bert_sim.restore(global_step=global_step)
	bert_sim.set_dict_terms(HPOReader().get_cns_list())

	if isinstance(src_terms, str):
		src_terms = open(src_terms).read().strip().splitlines()
	del_terms = get_del_words()
	src_terms = list(set([t.strip() for t in src_terms if len(t.strip()) > 0]))
	src_terms = [t for t in src_terms if t not in del_terms]

	tgtterm_score_pairs = bert_sim.predict_best_match(src_terms)
	samples = [(src_term, tgt_term, score) for src_term, (tgt_term, score) in zip(src_terms, tgtterm_score_pairs)]
	samples = sorted(samples, key=lambda item:item[2], reverse=True)
	write_standard_file(samples, save_txt, split_char=' | ')


def gen_jaccard_best_match(src_terms, save_txt, hpo_to_syn_terms, cpu_use=12):
	if isinstance(src_terms, str):
		src_terms = open(src_terms).read().strip().splitlines()
	dict_sim = DictSim('dict_sim', hpo_to_syn_terms=hpo_to_syn_terms, match_type='jaccard')
	tgtterm_score_pairs = dict_sim.predict_best(src_terms, cpu_use=cpu_use)
	samples = [(src_term, tgt_term, score) for src_term, (tgt_term, score) in zip(src_terms, tgtterm_score_pairs)]
	samples = sorted(samples, key=lambda item:item[2], reverse=True)
	write_standard_file(samples, save_txt, split_char=' | ')


def gen_word2hpos(src_terms, save_json, model_name, global_step=None,
		score_thres=None, match_type='all', cpu_use=12):
	"""old: = gen_best_match + gen_word2hpos_from_txt
	"""
	from bert_syn.core.bert_ddml_sim import BertDDMLSim
	from bert_syn.core.data_helper import HPOReader
	bert_sim = BertDDMLSim(model_name)
	bert_sim.restore(global_step=global_step)

	if isinstance(src_terms, str):
		src_terms = open(src_terms).read().strip().splitlines()
	del_terms = get_del_words()
	src_terms = list(set([t.strip() for t in src_terms if len(t.strip()) > 0]))
	src_terms = [t for t in src_terms if t not in del_terms]

	hpo_reader = HPOReader()
	matcher = AlbertDDMLMatcher(process_word2hpos(hpo_reader.get_cns_to_hpo()), bert_sim,
		score_thres=score_thres, match_type=match_type, cpu_use=cpu_use, stopwords=get_stopwords())
	results = matcher.match_many(src_terms)
	ret_word2hpos = {}
	for term, (hpos, scores) in zip(src_terms, results):
		if hpos:
			ret_word2hpos[term] = hpos
	os.makedirs(os.path.dirname(save_json), exist_ok=True)
	json.dump(ret_word2hpos, open(save_json, 'w'), indent=2, ensure_ascii=False)
	return ret_word2hpos


def gen_word2hpos_from_txt(txt_path, save_json=None, split_char='|', score_thres=None):
	lines = read_standard_file(txt_path, split_char=split_char)
	word2hpos = {} # {term: set([hpo1, hpo2, ...])}
	cns2hpos = HPOReader().get_cns_to_hpo()
	for src_term, hpo_term, score_str in lines:
		src_term, hpo_term = src_term.strip(), hpo_term.strip()
		score = float(score_str.strip())
		if score >= score_thres:
			dict_set_update(src_term, cns2hpos[hpo_term], word2hpos)
	os.makedirs(os.path.dirname(save_json), exist_ok=True)
	json.dump({word: list(hpos) for word, hpos in word2hpos.items()}, open(save_json, 'w'), indent=2, ensure_ascii=False)
	return word2hpos


def run_vocab_match(input_folder, output_folder, word2hpos, matcher='exact'):
	"""
	Args:
		word2hpos (dict or str): {term: set([hpo1, hpo2, ...])}
		match_type (str): 'exact' | 'bag'
	"""
	if isinstance(word2hpos, str):
		word2hpos = json.load(open(word2hpos))
	for word, hpos in list(word2hpos.items()):
		word2hpos[word] = set(hpos)

	hpo_reader = HPOReader()
	hpo_dict = hpo_reader.get_hpo_dict()
	hpo2text = {**{hpo: info['ENG_NAME'] for hpo, info in hpo_dict.items()}, **hpo_reader.get_hpo_to_cns()}

	matcher = ExactTermMatcher(word2codes=word2hpos) if matcher == 'exact' else BagTermMatcher(word2codes=word2hpos)
	searcher = MaxInvTextSearcher(matcher)

	input_jsons = sorted(get_file_list(input_folder, lambda p:p.endswith('.json')))
	for input_json in tqdm(input_jsons):
		output_json = input_json.replace(input_folder, output_folder)
		os.makedirs(os.path.dirname(output_json), exist_ok=True)
		field_to_info = json.load(open(input_json))
		for field, info in field_to_info.items():
			text = info['RAW_TEXT']
			search_result = searcher.search(text)
			entity_list = search_result_to_entity_list(text, search_result, hpo2text)
			info['ENTITY_LIST'] = entity_list
		json.dump(field_to_info, open(output_json, 'w'), indent=2, ensure_ascii=False)


def run_dict_bert_match(term_ranker,entity_method,input_folder, output_folder, word2hpos, model_name, global_step=None,
		score_thres=None, match_type='all', cpu_use=12, dict_matcher='exact', fchunk=10, all_input_jsons=None,
		jaccard_sym=True,):
	"""
	Args:
		term_ranker (str)
	"""
	if isinstance(word2hpos, str):
		word2hpos = json.load(open(word2hpos))
	for word, hpos in list(word2hpos.items()):
		word2hpos[word] = set(hpos)
	hpo_reader = HPOReader()
	hpo_dict = hpo_reader.get_hpo_dict()
	hpo2text = {**{hpo:info['ENG_NAME'] for hpo, info in hpo_dict.items()}, **hpo_reader.get_hpo_to_cns()}

	if term_ranker == 'bert':
		from bert_syn.core.bert_ddml_sim import BertDDMLSim
		bert_sim = BertDDMLSim(model_name)
		bert_sim.restore(global_step=global_step)
		term_ranker = AlbertDDMLMatcher(hpo_reader.get_cns_to_hpo(), bert_sim,
			score_thres=score_thres, match_type=match_type, cpu_use=cpu_use, stopwords=get_stopwords())

	elif term_ranker == 'jaccard':
		term_ranker = JaccardMatcher(hpo_reader.get_cns_to_hpo(), score_thres=score_thres,
			match_type=match_type, cpu_use=cpu_use, stopwords=get_stopwords(), jaccard_sym=jaccard_sym)
	else:
		assert False

	dict_matcher = ExactTermMatcher(word2codes=word2hpos) if dict_matcher == 'exact' else BagTermMatcher(word2codes=word2hpos)
	searcher = DictAndRankerTextSearcher(dict_matcher)


	if all_input_jsons is None:
		all_input_jsons = sorted(get_file_list(input_folder, lambda p:p.endswith('.json')))

	for input_idx in range(0, len(all_input_jsons), fchunk):

		input_jsons = all_input_jsons[input_idx: input_idx + fchunk]
		def cut_sentence_methods(input_jsons,entity_method):
			key_to_text = {}  # {(input_json, field_name): raw_text}
			if entity_method == 'unsupervised':
				for input_json in input_jsons:
					field_to_info = json.load(open(input_json))
					for field, info in field_to_info.items():
						key_to_text[(input_json, field)] = info['RAW_TEXT']
				keys, texts = zip(*key_to_text.items())
				return keys, texts,''
			elif entity_method =='NERorTAG':

				raw_texts = []
				for input_json in input_jsons:
					field_to_info = json.load(open(input_json))
					for field, info in field_to_info.items():
						#print('field,info',info)
						key_to_text[(input_json, field)] = info['ENTITY_LIST']
						raw_texts.append(info['RAW_TEXT'])
				keys, texts = zip(*key_to_text.items())
				return keys, texts,raw_texts

			else:
				assert False



		keys,texts,raw_texts = cut_sentence_methods(input_jsons,entity_method)






		search_results = searcher.search_multi_with_second_matcher(texts, term_ranker,entity_method,cpu_use=cpu_use)
		if entity_method == 'NERorTAG':
			texts = raw_texts

		outjson_to_dict = {}
		for (input_json, field), text, search_result in tqdm(zip(keys, texts, search_results)):

			entity_list = search_result_to_entity_list(text, search_result, hpo2text)
			output_json = input_json.replace(input_folder, output_folder)
			if output_json not in outjson_to_dict:
				outjson_to_dict[output_json] = {}
			outjson_to_dict[output_json][field] = {'RAW_TEXT':text, 'ENTITY_LIST':entity_list}
		for outjson, field_to_info in outjson_to_dict.items():
			os.makedirs(os.path.dirname(outjson), exist_ok=True)
			json.dump(field_to_info, open(outjson, 'w'), indent=2, ensure_ascii=False)
		print('processed json file: {}/{}'.format(min(input_idx + fchunk, len(all_input_jsons)), len(all_input_jsons)))


if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = '-1'



	model_name, global_step = 'hy_old_best_version_albertTinyDDMLSim-AN_Hpo_N_SD20_BGD20_PC_C0-3-1024-32-fc1024', 11000

	term_ranker,  jaccard_sym = 'bert',  False

	match_type,dict_matcher ='best', 'exact'

	score_thres_list = [-0.71,-0.80,-0.85,-0.91]
	for i_thres in score_thres_list:

		mark = 'hy_old_best_-' + str(global_step) +'-score_thres'+ '_'


		id = str(i_thres).split('.')[0][-1]+ str(i_thres).split('.')[1]
		#print(id)
		mark += id+'/'
		score_thres =i_thres



		input_folder = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/raw/PUMCH_Detection' \
					   '/277_cardiomyopathy/'

		output_folder = os.path.join('/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/bert_syn_project/data/',
									 'preprocess', 'PUCM_Gene_analysis/PUMCH_Detection/',mark)

		all_input_jsons = None


		word2hpos = HPOReader().get_cns_to_hpo()


		word2hpos = process_word2hpos(word2hpos)



		extract_entity_method = 'unsupervised'  # 'unsupervised'、'NERorTAG'

		run_dict_bert_match(term_ranker, extract_entity_method,input_folder, output_folder, word2hpos, model_name,
			global_step, score_thres, match_type, cpu_use=12, dict_matcher=dict_matcher, fchunk=500,
		all_input_jsons=all_input_jsons, jaccard_sym=jaccard_sym)


