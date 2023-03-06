
import chardet

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
		if self._is_num(c):
			return 1
		elif self._is_eng(c):		#
			return 2
		elif self._is_cns(c):		#
			return 3
		else:
			return 0

	def _is_num(self, c):
		return '\u0030' <= c <='\u0039'


	def _is_eng(self, c):
		return ('\u0041' <= c <='\u005a') or ('\u0061' <= c <='\u007a')


	def _is_cns(self, c):
		return '\u4e00' <= c <= '\u9fff'


if __name__ == '__main__':
	analyzer = StandardAnalyzer()
