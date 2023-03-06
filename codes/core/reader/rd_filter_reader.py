
from copy import deepcopy

from core.reader.rd_reader import RDReader
from core.utils.utils import check_return, get_all_ancestors_for_many

class RDFilterReader(object):
	def __init__(self, keep_rd_codes=None, keep_source_codes=None, keep_ances=False):
		"""
		Args:
			keep_rd_codes (iterable or None): None means keeping all rd_codes
			keep_source_codes (iterable or None): None means keeping all rd_codes related to given source_codes
			keep_ances (bool):
		"""
		super(RDFilterReader, self).__init__()
		self.rd_reader = RDReader()
		self.init_keep_rd_code_set(keep_rd_codes, keep_source_codes, keep_ances)
		self.FILTER_RD = self.keep_rd_code_set is not None and len(self.keep_rd_code_set) != 0

		self.rd_dict = None
		self.rd_list = None
		self.rd_map_rank = None
		self.source_code_to_rd_code = None
		self.rd_code_to_source_codes = None
		self.rd_num = None
		self.source_list = None


	def init_keep_rd_code_set(self, keep_rd_codes, keep_source_codes, keep_ances):
		if keep_rd_codes is not None:
			self.keep_rd_code_set = set(keep_rd_codes)
		elif keep_source_codes is not None:
			source_to_rd = self.rd_reader.get_source_to_rd()
			self.keep_rd_code_set = {source_to_rd[source_code] for source_code in keep_source_codes}
		else:
			self.keep_rd_code_set = None
		if self.keep_rd_code_set is not None and keep_ances:
			self.keep_rd_code_set.update(get_all_ancestors_for_many(self.keep_rd_code_set, self.rd_reader.get_rd_dict()))


	@check_return('rd_dict')
	def get_rd_dict(self):
		rd_dict = self.rd_reader.get_rd_dict()
		if not self.FILTER_RD:
			return rd_dict
		rd_dict = deepcopy(rd_dict)
		ret_rd_dict = {}
		for rd_code, info_dict in rd_dict.items():
			if rd_code not in self.keep_rd_code_set:
				continue
			info_dict['IS_A'] = [rd_code for rd_code in info_dict.get('IS_A', []) if rd_code in self.keep_rd_code_set]
			info_dict['CHILD'] = [rd_code for rd_code in info_dict.get('CHILD', []) if rd_code in self.keep_rd_code_set]
			ret_rd_dict[rd_code] = info_dict
		return ret_rd_dict


	@check_return('rd_list')
	def get_rd_list(self):
		"""
		Returns:
			dict: {
				DIS_CODE: {
					SOURCE_CODES: [ORPHA:XXX, OMIM:XXXXXX, ...],
					'LEVEL': str,
					'IS_A': [],
					'CHILD': [],
				}
			}
		"""
		rd_list = self.rd_reader.get_rd_list()
		if not self.FILTER_RD:
			return rd_list
		return [rd_code for rd_code in rd_list if rd_code in self.keep_rd_code_set]


	@check_return('rd_map_rank')
	def get_rd_map_rank(self):
		"""
		Returns:
			dict: {RD_CODE: int}
		"""
		if not self.FILTER_RD:
			return self.rd_reader.get_rd_map_rank()
		rd_list = self.get_rd_list()
		return {rd_code:i for i, rd_code in enumerate(rd_list)}


	@check_return('rd_num')
	def get_rd_num(self):
		if not self.FILTER_RD:
			return self.rd_reader.get_rd_num()
		return len(self.keep_rd_code_set)


	@check_return('source_code_to_rd_code')
	def get_source_to_rd(self):
		"""
		Returns:
			dict: {SOURCE_CODE: RD_CODE}
		"""
		source_to_rd = self.rd_reader.get_source_to_rd()
		if not self.FILTER_RD:
			return source_to_rd
		return {source: rd for source, rd in source_to_rd.items() if rd in self.keep_rd_code_set}


	@check_return('rd_code_to_source_codes')
	def get_rd_to_sources(self):
		"""
		Returns:
			dict: {RD_CODE: [SOURCE_CODE1, ...]}
		"""
		rd_dict = self.get_rd_dict()
		return {rd_code: rd_info['SOURCE_CODES'] for rd_code, rd_info in rd_dict.items()}


	def get_rd_to_level(self):
		rd_dict = self.get_rd_dict()
		return {rd_code: info['LEVEL'] for rd_code, info in rd_dict.items()}


	def get_source_to_level(self):
		rd_dict = self.get_rd_dict()
		return {source_code: info['LEVEL'] for _, info in rd_dict.items() for source_code in info['SOURCE_CODES']}


	@check_return('source_list')
	def get_source_list(self):
		return list(self.get_source_to_rd().keys())


	def statistic(self):
		from collections import Counter
		filtered_rd_dict = self.get_rd_dict()
		print('All filtered disease number: {}/{}'.format(len(filtered_rd_dict), len(all_source_codes)))  # 9216/11538
		print('Filtered disease level:', Counter([info['LEVEL'] for rd_code, info in
			filtered_rd_dict.items()]).most_common())  # [('DISORDER', 6306), ('DISORDER_SUBTYPE', 2619), ('DISORDER_GROUP', 291)]
		print('Filtered Isolated codes:', len([rd_code for rd_code, info in filtered_rd_dict.items() if
			len(info.get('IS_A', [])) == 0 and len(info.get('CHILD', [])) == 0]))


if __name__ == '__main__':
	from core.reader.hpo_filter_reader import HPOFilterDatasetReader
	all_source_codes = set(HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD']).get_dis_list())
	reader = RDFilterReader(keep_source_codes=all_source_codes, keep_ances=False)
	reader.statistic()




