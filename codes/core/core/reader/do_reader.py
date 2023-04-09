import json, pickle
import os
from tqdm import tqdm
import re
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from core.utils.constant import SPARSE_NPZ_FILE_FORMAT, JSON_FILE_FORMAT, PKL_FILE_FORMAT
from core.utils.constant import DATA_PATH, LOG_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, PHELIST_DESCENDENT, PHELIST_ORIGIN, PHELIST_ANCESTOR_DUP
from core.utils.utils import get_logger, dict_list_add, reverse_dict_list, delete_redundacy, read_standard_file, item_list_to_rank_list
from core.utils.utils import slice_dict_with_keep_set, check_load_save, check_load, get_all_ancestors_for_many, get_all_descendents_for_many, list_add_tail, get_all_dup_ancestors_for_many
from core.utils.utils import reverse_dict
from core.reader.obo_reader import OBOReader


class DOReader(object):
	def __init__(self):
		PREPROCESS_FOLDER = DATA_PATH + '/preprocess/knowledge/DO'
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)

		self.DOID_OBO = DATA_PATH + '/raw/DO/doid.obo'
		self.DO_DICT_JSON = PREPROCESS_FOLDER + '/DODict.json'
		self.DODict = None
		self.DO_SLICE_DICT_JSON = PREPROCESS_FOLDER + '/do_slice_dict.json'
		self.do_slice_dict = None

		self.DO_TO_OMIM_JSON = PREPROCESS_FOLDER + '/DoToOmim.json'
		self.do_to_omim = None
		self.OMIM_TO_DO_JSON = PREPROCESS_FOLDER + '/OmimToDo.json'
		self.omim_to_do = None


	@check_load_save('DODict', 'DO_DICT_JSON', JSON_FILE_FORMAT)
	def get_do_dict(self):
		"""
		Returns:
			dict: {doCode: info_dict}
		"""
		with open(self.DOID_OBO) as f:
			s = f.read()
			return OBOReader().loads(s[:s.find('[Typedef]')])


	@check_load_save('do_slice_dict', 'DO_SLICE_DICT_JSON', JSON_FILE_FORMAT)
	def get_do_slice_dict(self):
		do_dict = self.get_do_dict()
		return {code: slice_dict_with_keep_set(do_dict[code], {'IS_A', 'CHILD'}) for code in do_dict}


	@check_load_save('do_to_omim', 'DO_TO_OMIM_JSON', JSON_FILE_FORMAT)
	def get_do_to_omim(self):
		"""
		Returns:
			dict: {doCode: omim_code}
		"""
		ret_dict = {}
		do_dict = self.get_do_dict()
		for doCode, info in do_dict.items():
			for xref_code in info.get('XREF', []):
				if xref_code.startswith('OMIM'):
					assert xref_code not in ret_dict
					ret_dict[doCode] = xref_code
		return ret_dict


	@check_load_save('omim_to_do', 'OMIM_TO_DO_JSON', JSON_FILE_FORMAT)
	def get_omim_to_do_list(self):
		"""
		Returns:
			dict: {omim_code: [doCode, ...]}
		"""
		return reverse_dict(self.get_do_to_omim())


	def statistic(self):
		import random
		import json
		from core.reader.hpo_reader import HPOReader
		from core.explainer.explainer import Explainer
		from core.explainer.utils import add_info

		do_dict = self.get_do_dict()
		print('DO Code Number:', len(do_dict))   # 9233

		do_to_omim = self.get_do_to_omim()
		print('do_to_omim size:', len(do_to_omim))  # 3501

		omim_to_do_list = self.get_omim_to_do_list()
		print('omim_to_do_list size:', len(omim_to_do_list))  # 3462
		print('omim_to_do_list multiple do; size:', len([omim for omim, doList in omim_to_do_list.items() if len(doList) > 1]))   # 39

		hpo_reader = HPOReader()
		dis_list = hpo_reader.get_dis_list()
		hpo_omim_not_map_to_do = [dis_code for dis_code in dis_list if dis_code.startswith('OMIM') and dis_code not in omim_to_do_list]
		print('hpo_omim_not_map_to_do size:', len(hpo_omim_not_map_to_do))    # 4279
		print('hpo_omim_not_map_to_do sample:', random.sample(hpo_omim_not_map_to_do, 20))

		explainer = Explainer()
		do_dict = self.get_do_slice_dict()
		do_dict = add_info(do_dict, do_to_omim, (lambda tgt: isinstance(tgt, str)), 'w')
		do_dict = explainer.add_omim_info(do_dict)
		do_dict = explainer.add_do_info(do_dict)
		json.dump(do_dict, open(DATA_PATH + '/preprocess/DO-OMIM-CNS.json', 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':

	pass



