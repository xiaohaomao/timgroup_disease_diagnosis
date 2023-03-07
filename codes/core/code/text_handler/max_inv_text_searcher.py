
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
# 	text = """
# 主诉:行走不稳5年+
# 开出的检查项目:SCA1，2，3（同时取其妹乔树妹的血样）
# 分类:A1
# 姓名:乔书国
# 颅神经.眼震:凝视诱发眼震
# 颅神经.各方向眼动:突眼
# 初步诊断:SCA3？
# 病历号:1306669
# 祖籍:江苏南京
# 首诊日期:2011-6-22
# 性别:男
# 出生日期:1976-2-16
# 现病史:：5年+前出现行走不稳，言语欠清，偶有饮水发呛，眠可腰间盘突出，坐骨神经痛
# 一般情况.BP.立位:mmHg
# 一般情况.心率.立位:次/分
# 一般情况.智力:神清 构音障碍
# 治疗方案:巴氯芬，丁苯酞，辅酶Q10，VE
# 病历记录医师签名:顾卫红
# 运动系统.锥体系统.腱反射:亢进
# 运动系统.锥体系统.病理征:++
# 运动系统.锥体系统.肌张力:双上肢肌张力可，双下肢肌张力高
# 运动系统.脊髓小脑.跟膝胫:不稳
# 运动系统.脊髓小脑.指鼻:不稳
# 运动系统.脊髓小脑.快速轮替:笨拙
# 运动系统.脊髓小脑.步态:痉挛步态
# 	"""
# 	searcher = MaxInvTextSearcher(BagTermMatcher({}, ''))
# 	print(searcher.search(text))
# 	pass