import os
from tqdm import tqdm

from core.reader.hpo_reader import HPOReader
from core.utils.constant import DATA_PATH, PKL_FILE_FORMAT, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, reverse_dict, read_standard_file, check_return

class UMLSReader(object):
	def __init__(self):
		self.RAW_FOLDER = DATA_PATH + '/raw/UMLS'
		self.PREPROCESS_FOLDER = DATA_PATH + '/preprocess/knowledge/UMLS'
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)

		self.MRCONSO_RRF_PATH = self.RAW_FOLDER + '/MRCONSO.RRF'
		self.mrconso_dict = None #
		self.MRSTY_RRF_PATH = os.path.join(self.RAW_FOLDER, 'MRSTY.RRF')
		self.GRADED_MT_PKL = self.RAW_FOLDER + '/GradedMT_2017_1.1.pkl'
		self.graded_mt_dict = None
		self.SIMPLE_MT_PKL = self.RAW_FOLDER + '/SimpleMT_2017_1.1.pkl'
		self.simple_mt_dict = None
		self.AUI_TO_CUI_JSON = self.PREPROCESS_FOLDER + '/AUIToCUI.json'
		self.aui_to_cui_dict = None
		self.cui_to_aui_list = None
		self.HPO_TO_AUI_JSON = self.PREPROCESS_FOLDER + '/HPOToAUI.json'
		self.hpo_to_aui_dict = None
		self.AUI_TO_SAB_JSON = self.PREPROCESS_FOLDER + '/AUIToSAB.json'
		self.aui_to_sab = None
		self.CUI_TO_TUI_JSON = os.path.join(self.PREPROCESS_FOLDER, 'CUIToTUI.json')
		self.cui_to_tui = None
		self.tui_to_cuis = None
		self.TUI_TO_STY_JSON = os.path.join(self.PREPROCESS_FOLDER, 'TUIToSTY.json')
		self.tui_to_sty = None



	def get_mrconso(self):
		"""
		Returns:
			dict: {AUI1: line_dict1, AUI2: line_dict2, ...}, line_dict={colName1: value1, colName2: value2, ...}
		"""
		if self.mrconso_dict is not None:
			return self.mrconso_dict
		col_names = [
			'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
			'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
		]
		col_num = len(col_names)
		ret_dict = {}
		for line in tqdm(open(self.MRCONSO_RRF_PATH).readlines()):
			value_list = line.split('|')
			value_list.pop()
			line_dict = {col_names[i]:value_list[i] for i in range(col_num)}
			AUI = line_dict['AUI']
			assert AUI not in ret_dict
			ret_dict[AUI] = line_dict
		self.mrconso_dict = ret_dict
		return self.mrconso_dict


	def get_slice_mrconso(self, u_term_dictFilter, col_names):
		"""
		Args:
			u_term_dictFilter (func): args=(u_term_dict,); returns=(True or False,); u_term_dict={'CUI': cui, ...}; if True, then keep
			col_names (list): e.g. ['SAB', 'CUI']
		Returns:
			dict: {AUI: {colName1: value, ...}}
		"""
		mrconso_dict = self.get_mrconso()
		return {aui:  {col_name: u_term_dict[col_name] for col_name in col_names} for aui, u_term_dict in mrconso_dict.items() if u_term_dictFilter(u_term_dict)}


	@check_load_save('graded_mt_dict', 'GRADED_MT_PKL', PKL_FILE_FORMAT)
	def get_graded_mt(self):
		"""
		Returns:
			dict: {
				AUI: {
					'eng': xxx,
					'prefer': xxx,
					'preferSource': source,
					'confidence': int,
					'source': {source: cns}
				}
			}
		"""
		assert False


	@check_load_save('simple_mt_dict', 'SIMPLE_MT_PKL', PKL_FILE_FORMAT)
	def get_simple_mt(self):
		"""
		Returns:
			dict: {AUI: {'eng': eng, 'source': {source: cns}}}
		"""
		assert False


	@check_load_save('aui_to_cui_dict', 'AUI_TO_CUI_JSON', JSON_FILE_FORMAT)
	def get_aui_to_cui(self):
		"""
		Returns:
			dict: {AUI: CUI}
		"""
		mrconso_dict = self.get_mrconso()
		return {AUI: line_dict['CUI'] for AUI, line_dict in mrconso_dict.items()}


	def get_cui_to_aui_list(self):
		"""
		Returns:
			dict: {CUI: [AUI, ...]}
		"""
		if self.cui_to_aui_list is None:
			self.cui_to_aui_list = reverse_dict(self.get_aui_to_cui())
		return self.cui_to_aui_list


	def get_aui_syn_list(self, AUI):
		"""
		Returns:
			list: [AUI1, AUI2, ...]
		"""
		CUI = self.get_aui_to_cui()[AUI]
		return self.get_cui_to_aui_list()[CUI]


	@check_load_save('hpo_to_aui_dict', 'HPO_TO_AUI_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_aui(self):
		"""
		Returns:
			dict: {hpo_code: AUI}
		"""
		mrconso_dict = self.get_mrconso()
		hpo_set = set(HPOReader().get_hpo_list())
		return {line_dict['CODE']: AUI for AUI, line_dict in mrconso_dict.items() if line_dict['SAB'] == 'HPO' and line_dict['CODE'] in hpo_set}


	def get_hpo_to_cui(self):
		"""
		Returns:
			dict: {hpo_code: CUI}
		"""
		hpo_to_aui = self.get_hpo_to_aui()
		aui_to_cui = self.get_aui_to_cui()
		return {hpo: aui_to_cui[aui] for hpo, aui in hpo_to_aui.items()}


	def get_cui_to_hpo_list(self):
		"""
		Returns:
			dict: {CUI: [hpo_code1, ...]}
		"""
		return reverse_dict(self.get_hpo_to_cui())


	@check_load_save('aui_to_sab', 'AUI_TO_SAB_JSON', JSON_FILE_FORMAT)
	def get_aui_to_sab(self):
		"""
		Returns:
			dict: {AUI: SAB}
		"""
		mrconso_dict = self.get_mrconso()
		return {AUI: line_dict['SAB'] for AUI, line_dict in mrconso_dict.items()}


	@check_load_save('cui_to_tui', 'CUI_TO_TUI_JSON', JSON_FILE_FORMAT)
	def get_cui_to_tui(self):
		col_names = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF']
		name2col = {name: i for i, name in enumerate(col_names)}
		line_infos = read_standard_file(self.MRSTY_RRF_PATH, split_char='|')
		cui_to_tui = {}
		for line_info in line_infos:
			CUI = line_info[name2col['CUI']]
			TUI = line_info[name2col['TUI']]
			cui_to_tui[CUI] = TUI
		return cui_to_tui


	@check_return('tui_to_cuis')
	def get_tui_to_cuis(self):
		return reverse_dict(self.get_cui_to_tui())


	@check_load_save('tui_to_sty', 'TUI_TO_STY_JSON', JSON_FILE_FORMAT)
	def get_tui_to_sty(self):
		col_names = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF']
		name2col = {name:i for i, name in enumerate(col_names)}
		line_infos = read_standard_file(self.MRSTY_RRF_PATH, split_char='|')
		tui_to_sty = {}
		for line_info in line_infos:
			TUI = line_info[name2col['TUI']]
			STY = line_info[name2col['STY']]
			tui_to_sty[TUI] = STY
		sorted_keys = sorted(tui_to_sty.keys())
		return {k: tui_to_sty[k] for k in sorted_keys}


if __name__ == '__main__':
	umls_reader = UMLSReader()
	umls_reader.get_tui_to_sty()




