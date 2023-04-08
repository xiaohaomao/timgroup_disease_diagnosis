import os
import json
from tqdm import tqdm
import re
import itertools

from core.reader.umls_reader import UMLSReader
from core.reader.hpo_reader import HPOReader
from core.utils.constant import DATA_PATH, PKL_FILE_FORMAT, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, reverse_dict, remove_bracket, remove_begin_end, contain_punc, get_save_func
from core.utils.utils import contain_cns, all_cns, unique_list, get_all_descendents_for_many, jacard
from core.utils.constant import MT_HPO_SOURCE, MT_ICD10_SOURCE, MT_MESH_SOURCE, MT_SNOMED_SNMI_SOURCE
from core.utils.constant import MT_SNOMED_BDWK_SOURCE, MT_UMLS_CHI_SOURCE, MT_ICIBA_SOURCE
from core.analyzer.standard_analyzer import StandardAnalyzer


class SynGenerator(object):
	def __init__(self):
		self.umls_reader = UMLSReader()
		self.hpo_reader = HPOReader()
		self.stop_hpo_set = None


	def get_hpo_to_std_terms(self):
		"""
		Returns:
			dict: {hpo_code: [cns_term]}
		"""
		chpo_dict = self.hpo_reader.get_chpo_dict()
		stop_hpo_set = self.get_stop_hpo_set()
		return {hpo_code: [info_dict['CNS_NAME']] for hpo_code, info_dict in chpo_dict.items() if 'CNS_NAME' in info_dict and hpo_code not in stop_hpo_set}


	def get_stop_hpo_set(self):
		if self.stop_hpo_set is None:
			self.stop_hpo_set = get_all_descendents_for_many([
			'HP:0040279',
			'HP:0003679', 'HP:0003812', 'HP:0011008', 'HP:0012824', 'HP:0012830', 'HP:0025254', 'HP:0025280', 'HP:0025285', 'HP:0031375',
		], self.hpo_reader.get_slice_hpo_dict())
		return self.stop_hpo_set


class UMLSSynGenerator(SynGenerator):
	def __init__(self):
		super(UMLSSynGenerator, self).__init__()

		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'UMLS')
		self.HPO_TO_SYN_TERMS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'HPOToSynTerms.json')
		self.hpo_to_syn_terms = None
		self.HPO_TO_SOURCE_SYN_TERMS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'HPOToSourceSynTerms.json')
		self.hpo_to_source_syn_terms = None
		self.HPO_TO_BG_EVALUATE_SYN_TERMS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'HPOToBGEvaSynTerms.json')
		self.hpo_to_bg_eva_syn_terms = None
		self.HPO_TO_SYN_INFO_JSON = os.path.join(self.PREPROCESS_FOLDER, 'HPOToSynInfo.json')
		self.hpo_to_syn_info = None

		self.CUI_TO_SOURCE_SYN_TERMS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'CUIToSourceSynTerms.json')
		self.cui_to_source_syn_terms = None
		self.CUI_TO_BG_EVALUATE_SYN_TERMS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'CUIToBGEvaSynTerms.json')
		self.cui_to_bg_eva_syn_terms = None

		self.NOUSE_BRACKET_PATTERN = self.get_no_use_bracket_pattern()


	def get_no_use_bracket_pattern(self):
		lb = ['\[', '【', '\(', '（', '〔']
		rb = ['\]', '】', '\)', '）', '〕']
		bracket_terms = [
			'调查', '体格检查',
			'情况', '状况',
			'发现', '身体发现', '症状', '物理发现',
			'上下文相关类别', '上下文相关的类', '上下文依赖类',
			'病', '症', '病症', '病态', '疾病/寻找', '疾病/发现',
			'诊断', '障碍', '紊乱', '无效', '关键词', '\s*\w\s*', '生物功能', '寻找',
		]
		pattern = '[%s]{1}(%s){1}[%s]{1}' % (''.join(lb), '|'.join(bracket_terms), ''.join(rb))
		return re.compile(pattern, flags=re.A)


	def remove_no_use_bracket(self, term):
		return self.NOUSE_BRACKET_PATTERN.sub('', term)


	def cns_filter(self, cns_term):
		"""
		Returns:
			str or None:
		"""
		if type(cns_term) != str:
			return None
		cns_term = self.remove_no_use_bracket(cns_term)
		cns_term = remove_begin_end(cns_term)
		if not contain_cns(cns_term):
			return None
		if len(cns_term) < 2:
			return None
		return cns_term


	def do_nothing_mt_item_filter(self, mtItem):
		"""
		Returns:
			list: [cns_term1, cns_term2, ...]
		"""
		return [cns_term for source, cns_term in mtItem['source'].items()]


	def source_mt_item_filter(self, mtItem, keep_source_set=None):
		keep_source_set = keep_source_set or self.keep_source_set
		return [cns_term for source, cns_term in mtItem['source'].items() if source in keep_source_set]


	def evaluate_bg_trans_mt_item_filter(self, mtItem):
		source_dict = mtItem['source']
		if 'Baidu' not in source_dict or 'Google' not in source_dict:
			print('No Baidu or Google Translate:', source_dict)
			return [cns_term for source, cns_term in mtItem['source'].items() if source != 'Baidu' and source != 'Google']
		bTrans, g_trans = source_dict['Baidu'], source_dict['Google']

		ja = jacard(set(self.std_analyzer.split(bTrans)), set(self.std_analyzer.split(g_trans)))
		if ja >= self.min_ja:
			return [cns_term for source, cns_term in mtItem['source'].items()]
		else:
			return [cns_term for source, cns_term in mtItem['source'].items() if source != 'Baidu' and source != 'Google']


	@check_load_save('hpo_to_syn_info', 'HPO_TO_SYN_INFO_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_syn_info(self):
		"""
		Returns:
			dict: {hpo_code: {AUI: mtItem, ...}}
		"""
		hpo_to_aui = self.umls_reader.get_hpo_to_aui()
		aui_to_cui = self.umls_reader.get_aui_to_cui()
		cui_to_aui_list = self.umls_reader.get_cui_to_aui_list()
		mt_dict = self.umls_reader.get_simple_mt()
		ret_dict = {}
		for hpo_code, AUI in tqdm(hpo_to_aui.items()):
			hpo_cui = aui_to_cui[AUI]
			syn_auis = cui_to_aui_list[hpo_cui]
			ret_dict[hpo_code] = {synAUI: mt_dict[synAUI] for synAUI in syn_auis if synAUI in mt_dict}
		aui_to_sab = self.umls_reader.get_aui_to_sab()
		for hpo_code in tqdm(ret_dict):
			for AUI in ret_dict[hpo_code]:
				ret_dict[hpo_code][AUI]['SAB'] = aui_to_sab[AUI]
		return ret_dict


	def get_hpo_to_syn_terms_base(self, mtItemFilter, cns_filter):
		"""
		Args:
			mtItemFilter (func): args=(mtItem,), returns=(cnsList,)
			cns_filter (func): args=(cns_term,), returns=(cns_term or None,)
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		hpo_to_aui = self.umls_reader.get_hpo_to_aui()
		aui_to_cui = self.umls_reader.get_aui_to_cui()
		cui_to_aui_list = self.umls_reader.get_cui_to_aui_list()
		mt_dict = self.umls_reader.get_simple_mt()
		ret_dict = {}
		for hpo_code, AUI in tqdm(hpo_to_aui.items()):
			hpo_cui = aui_to_cui[AUI]
			syn_auis = cui_to_aui_list[hpo_cui]
			syn_mt_items = [mt_dict[synAUI] for synAUI in syn_auis if synAUI in mt_dict]
			cns_terms = []
			for mtItem in syn_mt_items:
				cns_terms.extend(mtItemFilter(mtItem))
			filtered_cns_terms = []
			for cns_term in cns_terms:
				filtered_cns = cns_filter(cns_term)
				if filtered_cns is not None:
					filtered_cns_terms.append(filtered_cns)
			ret_dict[hpo_code] = unique_list(filtered_cns_terms)
		chpo_dict = self.hpo_reader.get_chpo_dict()
		for hpo_code, infoItem in chpo_dict.items():
			if hpo_code not in ret_dict:
				print('========== hpo_code ======',hpo_code)
				print('========== infoItem =======',infoItem)
				ret_dict[hpo_code] = [infoItem['CNS_NAME']]
		ret_dict = {hpo_code: cns_terms for hpo_code, cns_terms in ret_dict.items() if hpo_code not in self.get_stop_hpo_set()}
		return ret_dict


	@check_load_save('hpo_to_syn_terms', 'HPO_TO_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_syn_terms(self):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		return self.get_hpo_to_syn_terms_base(self.do_nothing_mt_item_filter, self.cns_filter)


	@check_load_save('hpo_to_source_syn_terms', 'HPO_TO_SOURCE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_source_syn_terms(self):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		self.keep_source_set = {
			MT_HPO_SOURCE, MT_ICD10_SOURCE, MT_MESH_SOURCE, MT_SNOMED_SNMI_SOURCE,
			MT_SNOMED_BDWK_SOURCE, MT_UMLS_CHI_SOURCE, MT_ICIBA_SOURCE
		}
		ret_dict = self.get_hpo_to_syn_terms_base(self.source_mt_item_filter, self.cns_filter)
		return ret_dict


	@check_load_save('hpo_to_bg_eva_syn_terms', 'HPO_TO_BG_EVALUATE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_syn_terms_with_bg_evaluate(self, min_ja=1.0):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		self.min_ja = min_ja
		self.std_analyzer = StandardAnalyzer()
		ret_dict = self.get_hpo_to_syn_terms_base(self.evaluate_bg_trans_mt_item_filter, self.cns_filter)
		return ret_dict


	def get_cui_syn_dict_base(self, mtItemFilter, cns_filter):
		"""
		Args:
			save_path (str): .json or .pkl; {CUI: term}
		"""
		import numpy as np
		cui_to_aui_list = self.umls_reader.get_cui_to_aui_list()
		mt_dict = self.umls_reader.get_simple_mt()
		cui_to_syns = {}
		for CUI, syn_auis in tqdm(cui_to_aui_list.items()):
			syn_mt_items = [mt_dict[synAUI] for synAUI in syn_auis if synAUI in mt_dict]
			cns_terms = []
			for mtItem in syn_mt_items:
				cns_terms.extend(mtItemFilter(mtItem))
			filtered_cns_terms = []
			for cns_term in cns_terms:
				filtered_cns = cns_filter(cns_term)
				if filtered_cns is not None:
					filtered_cns_terms.append(filtered_cns)
			filtered_cns_terms = unique_list(filtered_cns_terms)
			if len(filtered_cns_terms) > 0:
				cui_to_syns[CUI] = filtered_cns_terms
		return cui_to_syns


	@check_load_save('cui_to_source_syn_terms', 'CUI_TO_SOURCE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_cui_to_source_syn_terms(self):
		self.keep_source_set = {
			MT_HPO_SOURCE, MT_ICD10_SOURCE, MT_MESH_SOURCE, MT_SNOMED_SNMI_SOURCE,
			MT_SNOMED_BDWK_SOURCE, MT_UMLS_CHI_SOURCE, MT_ICIBA_SOURCE
		}
		return self.get_cui_syn_dict_base(self.source_mt_item_filter, self.cns_filter)


	@check_load_save('cui_to_bg_eva_syn_terms', 'CUI_TO_BG_EVALUATE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_cui_to_syn_terms_with_bg_evaluate(self, min_ja=1.0):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		self.min_ja = min_ja
		self.std_analyzer = StandardAnalyzer()
		ret_dict = self.get_cui_syn_dict_base(self.evaluate_bg_trans_mt_item_filter, self.cns_filter)
		return ret_dict


if __name__ == '__main__':
	sg = UMLSSynGenerator()
	sg.get_hpo_to_syn_info()
	sg.get_hpo_to_syn_terms()
	sg.get_hpo_to_source_syn_terms()
	sg.get_hpo_to_syn_terms_with_bg_evaluate()

	sg.get_cui_to_source_syn_terms()
	sg.get_cui_to_syn_terms_with_bg_evaluate()
