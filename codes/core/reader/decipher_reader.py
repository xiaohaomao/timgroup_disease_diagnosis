

import os
from tqdm import tqdm

from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import check_load_save

class DecipherReader(object):
	def __init__(self):
		PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'Decipher')
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)

		self.DECIPHER_DICT_JSON = os.path.join(PREPROCESS_FOLDER, 'decipher_dict.json')
		self.decipher_dict = None


	@check_load_save('decipher_dict', 'DECIPHER_DICT_JSON', JSON_FILE_FORMAT)
	def get_decipher_dict(self):
		"""
		Returns:
			dict: {
				decipherCode: {
					'ENG_NAME': '',
				}
			}
		"""
		from core.reader.hpo_reader import HPOReader
		ret_dict = {}
		col_names, info_lists = HPOReader().read_phenotype_anno_hpoa(); COL_NUM = len(col_names)
		name2rank = {name:i for i, name in enumerate(col_names)}
		for info_list in tqdm(info_lists):
			dis_code = info_list[name2rank['DATABASE_ID']]
			if dis_code.startswith('DECIPHER'):
				eng_name = info_list[name2rank['DISEASE_NAME']]
				ret_dict[dis_code] = {'ENG_NAME': eng_name}
		return ret_dict


if __name__ == '__main__':
	reader = DecipherReader()
	reader.get_decipher_dict()
