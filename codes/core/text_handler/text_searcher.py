
import re
from core.utils.utils import merge_pos_list, unique_list


class TextSearcher(object):
	def __init__(self):
		self.SPLIT_SEN_PATTEN = re.compile(r'[，。！？；\n]')
		self.neg_terms = {
			'未引出', '未见', '没有', '未有', '否认', '无', '（-）',
			'不明显', '未再', '未出现', '不符合', '不考虑', '未诉', '未见异常', '不伴', '除外'
		}
		self.stop_words = {'有时', '的', '较前', '稍'}


	def _search_sentence(self, text):

		raise NotImplementedError


	def set_sen_split_pattern(self, pattern):
		self.SPLIT_SEN_PATTEN = pattern


	def set_neg_terms(self, neg_terms):
		self.neg_terms = neg_terms


	def add_neg_terms(self, neg_terms):
		self.neg_terms.update(neg_terms)


	def del_neg_terms(self, neg_terms):
		for neg_term in neg_terms:
			self.neg_terms.remove(neg_term)


	def set_split_sen_pattern(self, splitPattern):
		self.SPLIT_SEN_PATTEN = re.compile(splitPattern)


	def set_stop_words(self, stop_words):
		self.stop_words = set(stop_words)


	def add_stop_words(self, stop_words):
		self.stop_words.update(stop_words)


	def search(self, doc):
		"""
		Args:
			doc (str)
		Returns:
			list: hpo_list; [hpo_code1, hpo_code2, ...]
			list: pos_list; [np.array([begin1, end1]), ...]
		"""
		ret_hpo_list, ret_pos_list = [], []
		sen_list, offset_list = self.doc_to_sen_list(doc)
		for i in range(len(sen_list)):
			hpo_list, pos_list = self._search_sentence(sen_list[i])
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
		restStr, p = '', 0
		for b, e in mpos_list:
			restStr += sentence[p: b]
			p = e
		restStr += sentence[p: len(sentence)]
		for negTerm in self.neg_terms:
			if restStr.find(negTerm) > -1:
				return [], []
		return match_hpos, pos_list


	def sort_result(self, match_hpos, pos_list):
		if len(match_hpos) == 0:
			return match_hpos, pos_list
		match_hpos, pos_list = zip(*sorted(zip(match_hpos, pos_list), key=lambda item: item[1][0]))
		return list(match_hpos), list(pos_list)


	def doc_to_sen_list(self, doc):
		"""
		Returns:
			list: list of sentences
			list: [sen1Begin, sen2Begin, ...], offset of sentence
		"""
		sen_list = self.SPLIT_SEN_PATTEN.split(doc)
		offset_list = [0]
		for sen in sen_list:
			offset_list.append(offset_list[-1]+len(sen)+1)
		offset_list.pop()
		return sen_list, offset_list



if __name__ == '__main__':
	pass









