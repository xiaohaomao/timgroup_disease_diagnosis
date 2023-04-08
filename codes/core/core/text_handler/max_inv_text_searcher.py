
import numpy as np
from core.text_handler.text_searcher import TextSearcher
from core.utils.utils import is_punc, is_space

class MaxInvTextSearcher(TextSearcher):
	def __init__(self, term_matcher, invalid_end_set=None, skip_term=None):
		super(MaxInvTextSearcher, self).__init__()
		self.matcher = term_matcher
		self.max_term_length = term_matcher.get_max_term_length()
		self.name = '{}-{}'.format('MaxInvTextSearcher', self.matcher.name)

		self.invalid_end_set = {} if invalid_end_set is None else invalid_end_set
		self.skip_term = {} if skip_term is None else skip_term


	def _search_sentence(self, text):

		info_list, pos_list = [], []
		end = len(text)
		while end != 0:
			for begin in range(max(0, end-self.max_term_length), end):
				term = text[begin: end]
				if term in self.skip_term:
					end = begin + 1
					break
				if self.invalid_begin_end(term):
					continue
				match_list, _ = self.matcher.match(term)
				if match_list:
					info_list.extend(match_list)
					pos_list.extend([np.array([begin, end]) for _ in match_list])
					end = begin + 1
					break
			end -= 1
		return info_list, pos_list


	def invalid_begin_end(self, term):
		return len(term) == 0 or self.invalid_begin(term[0]) or self.invalid_end(term[-1])


	def invalid_begin(self, c):
		return is_punc(c) or is_space(c)


	def invalid_end(self, c):
		if c in self.invalid_end_set:
			return False
		return is_punc(c) or is_space(c)


if __name__ == '__main__':
	from core.text_handler.term_matcher import BagTermMatcher
