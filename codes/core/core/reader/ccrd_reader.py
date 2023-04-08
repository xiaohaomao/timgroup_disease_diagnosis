import os
import json

from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, check_load, check_return, dict_list_add
from core.reader.hpo_reader import HPOReader

class CCRDReader(object):
	def __init__(self):
		self.link2reversed = {'NTBT':'BTNT', 'BTNT':'NTBT', 'NTBT/E':'BTNT/E', 'BTNT/E':'NTBT/E'}
		self.RAW_CCRD_JSON_PATH = os.path.join(DATA_PATH, 'raw', 'CCRD', 'conpendium_hpo.json')
		self.PROCESS_CCRD_JSON_PATH = os.path.join(DATA_PATH, 'raw', 'CCRD', 'conpendium_hpo_process.json')
		self.ccrd_dict = None

		self.CCRD_TO_ORPHA = os.path.join(DATA_PATH, 'raw', 'CCRD', 'ccrd_to_orpha.json')
		self.ccrd_to_orpha = None
		self.orpha_to_ccrd = None
		self.CCRD_TO_OMIM = os.path.join(DATA_PATH, 'raw', 'CCRD', 'ccrd_to_omim.json')
		self.ccrd_to_omim = None
		self.omim_to_ccrd = None


	@check_load_save('ccrd_dict', 'PROCESS_CCRD_JSON_PATH', JSON_FILE_FORMAT)
	def get_ccrd_dict(self):
		"""
		Returns:
			dict: {ccrd_code: {'CNS_NAME': str, 'ENG_NAME': str, 'PHENOTYPE_LIST': [hpo_code, ...]}}
		"""
		hpo_reader = HPOReader()
		hpo_old2new = hpo_reader.get_old_map_new_hpo_dict()
		hpo_dict = hpo_reader.get_hpo_dict()
		ccrd2info = json.load(open(self.RAW_CCRD_JSON_PATH))
		for ccrd_code, info in ccrd2info.items():
			info['PHENOTYPE_LIST'] = self.hpo_list_old_to_new(info['PHENOTYPE_LIST'], hpo_dict, hpo_old2new)
		return ccrd2info




	def hpo_list_old_to_new(self, hpo_list, hpo_dict, old2new):
		"""old->new
		"""
		new_hpo_list = []
		for hpo_code in hpo_list:
			if hpo_code not in hpo_dict:
				if hpo_code in old2new:
					new_code = old2new[hpo_code]
					new_hpo_list.append(new_code)
					print('{}(old) -> {}(new)'.format(hpo_code, new_code))
				else:
					print('delete {}'.format(hpo_code))
			else:
				new_hpo_list.append(hpo_code)
		return new_hpo_list


	def reverse_source_mapping(self, s1_to_s2):
		"""
		Args:
			s1_to_s2 (dict): {S1_CODE: [(S2_CODE, S1_TO_S2), ...]}
		Returns:
			dict: {S2_CODE: [(S1_CODE, S2_TO_S1), ...]}
		"""
		ret_dict = {}
		for s1_code, s2_list in s1_to_s2.items():
			for s2_code, link_s1_to_s2 in s2_list:
				link_s2_to_s1 = self.link2reversed.get(link_s1_to_s2, '') or link_s1_to_s2
				dict_list_add(s2_code, (s1_code, link_s2_to_s1), ret_dict)
		return ret_dict


	@check_load('ccrd_to_orpha', 'CCRD_TO_ORPHA', JSON_FILE_FORMAT)
	def get_ccrd_to_orpha(self):
		"""
		Returns:
			dict: {CCRD_CODE: [(ORPHA_CODE, CCRD_TO_ORPHA), ...]}
		"""
		pass


	@check_return('orpha_to_ccrd')
	def get_orpha_to_ccrd(self):
		"""
		Returns:
			dict: {ORPHA_CODE: [(CCRD_CODE, ORPHA_TO_CCRD), ...]}
		"""
		return self.reverse_source_mapping(self.get_ccrd_to_orpha())


	@check_load('ccrd_to_omim', 'CCRD_TO_OMIM', JSON_FILE_FORMAT)
	def get_ccrd_to_omim(self):
		"""
		Returns:
			dict: {CCRD_CODE: [(OMIM_CODE, CCRD_TO_OMIM), ...]}
		"""
		pass


	@check_return('omim_to_ccrd')
	def get_omim_to_ccrd(self):
		"""
		Returns:
			dict: {OMIM_CODE: [(OMIM_CODE, OMIM_TO_CCRD), ...]}
		"""
		return self.reverse_source_mapping(self.get_ccrd_to_omim())



if __name__ == '__main__':
	ccrd_reader = CCRDReader()
	ccrd_dict = ccrd_reader.get_ccrd_dict()



