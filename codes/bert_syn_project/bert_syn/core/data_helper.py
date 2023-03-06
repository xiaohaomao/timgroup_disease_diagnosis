
import os
import re
import json
import pickle
import pandas as pd
import numpy as np
import random
import itertools
from tqdm import tqdm
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from bert_syn.utils.constant import DATA_PATH, RESULT_PATH, PKL_FILE_FORMAT, JSON_FILE_FORMAT, TEST_DATA, VALIDATION_DATA
from bert_syn.utils.utils import check_load, check_return, reverse_dict, cal_jaccard_sim_list, timer, check_load_save, get_all_ancestors
from bert_syn.utils.utils import dict_set_add, unique_pairs, cal_quartile, contain_cns, dict_set_add, dict_set_update, equal_to, unique_lists
from bert_syn.utils.utils import get_load_func, get_save_func, get_all_descendents_for_many, reverse_dict_list
from bert_syn.utils.utils_draw import simple_dist_plot
from bert_syn.core.explainer import SampleExplainer




def get_del_words():
	del_words = {
		'甲状腺', '直肠', '前间隙', '盆腔', '积水', '关节', '心脏', '双下肢', '双肺', '双肺下叶', '胸腰椎', '尿管', '双肾上腺',
		'睡眠', '肾功能', '右心', '腹壁', '腹腔', '肾功', '呼吸', '运动', '右心', '双眼视力', '嗅觉', '视力', '双足',
		'线粒体', '白细胞', '红细胞',
		'免疫', '免疫组化', '尿常规', '血常规',
		'查蛋白尿', '尿蛋白', '尿ACR', '患者精神', '精神', '睡眠可', '患者病来精神', '出血', '红肿',
		'发作', '发红', '发白', '方相', '脱落', '开口', '缺乏', '保留', '杂音', '前开', '重复', '撕裂', '一个', '外翻', '摇摆', '呼吸音', # from syn_source+bg
		'患者', '治疗', '处理', '反复', '范围', '凝血', '出牙', '管的', '刺激', '无力', '突出', '麻醉', '关闭', '反折', '闭塞', # from syn_source_bg_dict
		'现病史', '既往史', '入院情况', '入院诊断', '出院诊断', '家族史', '病案号', '姓名', '既往体健'
	}
	del_words.update(SCELReader().get_multi_vocab([
		'人体解剖学名词（中文）2009', '人体组织名称', '解剖学词库', '人体解剖学名词【官方推荐】',
		'医疗器械大全【官方推荐】', '临床常用药物',
		'《医疗机构临床检验项目目录》（2013年版）', '医学检验',
		'国际标准手术编码',
		'分子学检验学', '分子遗传学常用词汇', '生物信息学常用词汇',
		'微生物学常用词汇', '细胞生物学词库',
		# '2017国家医保目录西药药品', '临床用药大全STZ', '药品', '药品对照品', '药品名称+商品名', '药品名称大全', '药学词库', '医药名称', '医院西药名称',
	]))

	keep_words = HPOReader().get_cns_list()
	keep_words.remove('异位'); keep_words.remove('缺损') # from CHPO
	keep_words.extend([
		'骨质疏松症', '小骨盆', '胸水', '小舌', '脚长', '自身免疫疾病', '黄斑',
		'长脚', '桡骨小头半脱位', '胆石', '骨刺', '宽胸', '过敏反应'])
	for w in keep_words:
		if w in del_words:
			del_words.remove(w)
	return del_words


def get_del_hpos():
	return get_all_descendents_for_many(
		['HP:0012823', 'HP:0031797', 'HP:0032223', 'HP:0032443', 'HP:0040279'], HPOReader().get_hpo_dict())


def get_del_subwords():
	return {
		'进一步诊治', '体健', '呼吸音清', '就诊我院'
	}


def get_stopwords():
	return [
		'入院', '出院', '医院', '我院', '我科', '病房', '收入', '收入院', '门诊', '朝阳医院', '协和医院', '空军总医院',
		'诊断', '诊治', '就诊', '复诊', '治疗', '查体', '患者',
		'提示', '出现', '立即', '发现', '考虑','仍诉', '发生', '表现为', '表现出',
		'同时', '逐渐', '此后', '其后', '近期', '开始', '进一步',
		'病程', '病程中', '症状', '上述', '为求', '同前述', '既往史',
		'给予', '应用',  '完善',  '转至', '监测', '建议', '重视',
	]


class UMLSReader():
	def __init__(self):

		self.SAVE_FOLDER = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/preprocess/knowledge/UMLS'


		self.SIMPLE_MT_PKL = os.path.join(self.SAVE_FOLDER, 'SimpleMT_2017_1.1.pkl')
		self.simple_mt_dict = None
		self.AUI_TO_CUI_JSON = os.path.join(self.SAVE_FOLDER, 'AUIToCUI.json')
		self.aui_to_cui_dict = None
		self.cui_to_aui_list = None
		self.HPO_TO_AUI_JSON = os.path.join(self.SAVE_FOLDER, 'HPOToAUI.json')
		self.hpo_to_aui_dict = None
		self.AUI_TO_SAB_JSON = os.path.join(self.SAVE_FOLDER, 'AUIToSAB.json')
		self.aui_to_sab = None
		self.CUI_TO_TUI_JSON = os.path.join(self.SAVE_FOLDER, 'CUIToTUI.json')
		self.cui_to_tui = None
		self.tui_to_cuis = None
		self.TUI_TO_STY_JSON = os.path.join(self.SAVE_FOLDER, 'TUIToSTY.json')
		self.tui_to_sty = None

		self.MT_SAVE_FOLDER = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/raw/TranslateUMLS'
		self.AUI_LISTS_JSON = os.path.join(self.MT_SAVE_FOLDER, 'AUI.json')
		self.aui_lists = None
		self.ENG_LIST_JSON = os.path.join(self.MT_SAVE_FOLDER, 'termENG.json')
		self.eng_list = None

		self.hpo_to_cui = None
		self.cui_to_hpo_list = None


	@check_load('simple_mt_dict', 'SIMPLE_MT_PKL', PKL_FILE_FORMAT)
	def get_simple_mt(self):
		"""
		Returns:
			dict: {AUI: {'eng': eng, 'source': {source: cns}}}
		"""
		assert False


	@check_load('aui_to_cui_dict', 'AUI_TO_CUI_JSON', JSON_FILE_FORMAT)
	def get_aui_to_cui(self):
		"""
		Returns:
			dict: {AUI: CUI}
		"""
		assert False


	@check_return('cui_to_aui_list')
	def get_cui_to_aui_list(self):
		"""
		Returns:
			dict: {CUI: [AUI, ...]}
		"""
		return reverse_dict(self.get_aui_to_cui())


	@check_load('hpo_to_aui_dict', 'HPO_TO_AUI_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_aui(self):
		"""
		Returns:
			dict: {hpo_code: AUI}
		"""
		assert False


	@check_return('hpo_to_cui')
	def get_hpo_to_cui(self):
		"""
		Returns:
			dict: {hpo_code: CUI}
		"""
		hpo_to_aui = self.get_hpo_to_aui()
		aui_to_cui = self.get_aui_to_cui()
		return {hpo:aui_to_cui[aui] for hpo, aui in hpo_to_aui.items()}


	@check_return('cui_to_hpo_list')
	def get_cui_to_hpo_list(self):
		"""
		Returns:
			dict: {CUI: [hpo_code1, ...]}
		"""
		return reverse_dict(self.get_hpo_to_cui())


	@check_load('aui_to_sab', 'AUI_TO_SAB_JSON', JSON_FILE_FORMAT)
	def get_aui_to_sab(self):
		"""
		Returns:
			dict: {AUI: SAB}
		"""
		assert False


	@check_load('aui_lists', 'AUI_LISTS_JSON', JSON_FILE_FORMAT)
	def get_aui_lists(self):
		"""
		Returns:
			list: [[AUI1, AUI2, ...]], AUI in the same list map to the same
		"""
		assert False


	@check_load('eng_list', 'ENG_LIST_JSON', JSON_FILE_FORMAT)
	def get_eng_list(self):
		"""
		Returns:
			list: [eng_str1, eng_str2];
		"""
		assert False


	@check_load('cui_to_tui', 'CUI_TO_TUI_JSON', JSON_FILE_FORMAT)
	def get_cui_to_tui(self):
		"""
		Returns:
			dict: {CUI: TUI}
		"""
		assert False


	@check_load_save('tui_to_sty', 'TUI_TO_STY_JSON', JSON_FILE_FORMAT)
	def get_tui_to_sty(self):
		"""
		Returns:
			dict: {TUI: STY_NAME}
		"""
		assert False


	@check_return('tui_to_cuis')
	def get_tui_to_cuis(self):
		return reverse_dict(self.get_cui_to_tui())


	def get_cuis_given_tuis(self, tuis):
		"""
		Args:
			tuis (list): [tui1, tui2, ...]
		Returns:
			list: [cui1, cui2, ...]
		"""
		tui_to_cuis = self.get_tui_to_cuis()
		ret_cuis = []
		for tui in tuis:
			ret_cuis.extend(tui_to_cuis[tui])
		return list(set(ret_cuis))


	def get_terms_given_tuis(self, cui_to_syns, tuis):
		cuis = self.get_cuis_given_tuis(tuis)
		ret_syns = []
		for cui in cuis:
			ret_syns.extend(cui_to_syns.get(cui, []))
		return list(set(ret_syns))


class HPOReader(object):
	def __init__(self):

		self.SAVE_FOLDER = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/preprocess/knowledge/HPO'

		self.CHPO_JSON_PATH = os.path.join(self.SAVE_FOLDER, 'chpo_dict.json')
		self.HPO_JSON_PATH = os.path.join(self.SAVE_FOLDER, 'hpo_dict.json')
		self.OLDMAPNEWHPO_JSON_PATH = os.path.join(self.SAVE_FOLDER, 'old_map_new_hpo.json')
		self.hpo_dict = None
		self.chpo_dict = None
		self.cns_to_hpo = None
		self.hpo_to_cns = None
		self.cns_list = None
		self.old_map_new_hpo = None  # {OLD_HPO_CODE: NEW_HPO_CODE}


	@check_load('hpo_dict', 'HPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_hpo_dict(self):
		"""
		Returns:
			dict: {hpo_code: {
			'ENG_NAME': '英文名', 'IS_A': ['父节点的hpo code'], 'CHILD': ['子节点的hpo code'], 'ENG_DEF': '定义', 'SYNONYM': ['同义词'], 'CREATED_DATE': '创建时间',
			'CREATED_BY': '创建者', 'ALT_ID': [], 'XREF': [], 'COMMENT': 'comment', 'replaced_by': 'REPLACED_BY'}}
		"""
		assert False


	@check_load('chpo_dict', 'CHPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_chpo_dict(self):
		"""
		Returns:
			dict: {
				hpo_code: {
					'MAIN_CLASS': str,
					'ENG_NAME': str,
					'CNS_NAME': str,
					'ENG_DEF': str,
					'CNS_DEF': str
					}, ...
				}
		"""
		assert False


	@check_return('hpo_to_cns')
	def get_hpo_to_cns(self):
		"""
		Returns:
			dict: {hpo_code: cns_term}
		"""
		chpo_dict = self.get_chpo_dict()
		return {hpo:info['CNS_NAME'] for hpo, info in chpo_dict.items() if info.get('CNS_NAME', '')}


	@check_return('cns_to_hpo')
	def get_cns_to_hpo(self):
		"""
		Returns:
			dict: {cns_term: [hpo_code1, ...]}
		"""
		return reverse_dict(self.get_hpo_to_cns())


	@check_return('cns_list')
	def get_cns_list(self):
		return sorted(self.get_cns_to_hpo().keys())


	def get_hpo_to_eng(self, lower=False):
		"""
		Returns:
			dict: {hpo_code: [eng_term1, ...]}
		"""
		hpo_dict = self.get_hpo_dict()
		ret_dict = {}
		for hpo, info in hpo_dict.items():
			eng_list = [info['ENG_NAME']] + info.get('SYNONYM', [])
			if lower:
				eng_list = [term.lower() for term in eng_list if not term.isupper()]
			ret_dict[hpo] = eng_list
		return ret_dict


	def get_eng_to_hpo(self, lower=False):
		"""
		Returns:
			dict: {eng_term: [hpo_code1, ...]}
		"""
		return reverse_dict_list(self.get_hpo_to_eng(lower))


	@check_load_save('old_map_new_hpo', 'OLDMAPNEWHPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_old_map_new_hpo_dict(self):
		"""
		Returns:
			dict: {old_hpo: new_hpo}
		"""
		assert False


	def statistics(self):
		print('HPO term number:', len(self.get_chpo_dict()))  # 11703

		save_folder = os.path.join(RESULT_PATH, 'hpo_statistics'); os.makedirs(save_folder, exist_ok=True)
		cns_list = self.get_cns_list()
		len_list = [len(term) for term in cns_list]
		print('Length quartile:', cal_quartile(len_list))
		simple_dist_plot(os.path.join(save_folder, 'len_dist.png'), len_list, bins=20, x_label='Text length', title='Text length dist')

		cns_list = self.get_cns_list()
		for t in cns_list:
			assert len(t.strip()) != 0


	def get_antonyms_pairs(self):
		return [
			(['上升', '升高', '增加', '增强', '增多'], ['下降', '降低', '减少', '减退', '减轻', '低下', '缺乏', '减弱']),
			(['正常', '如常', '稳定', '未见异常', '尚可', '可', '良好'], ['不规律', '不规则', '异常', '不良', '障碍', '不全', '不稳', '不能', '失常', '病', '畸形']),
			(['扩张', '扩大', '增大', '宽', '肥大', '肿大', '增生'], ['缩窄', '狭窄', '缩小']),
			(['亢进', '活跃'], ['减弱', '减低', '消失']),
			(['相等', '等大'], ['不等']),
			(['大', '不小'], ['小', '不大']),
			(['多', '不少'], ['少', '不多']),
			(['高', '不低'], ['低', '不高']),
			(['对称'], ['不对称']),
			(['自主'], ['不自主']),
			(['整齐'], ['不齐']),
			(['平衡'], ['不平衡']),
			(['未'], ['正常', '已']),
			(['大于'], ['小于']),
			(['阳性', '(+)', '（+）'], ['阴性', '([-－])', '（[-－]）']),
			(['音清'], ['杂音', '音异常', '音粗']),
			(['规律', '正常'], ['频繁', '不规律', '次数增多']),
			(['过度', '过分', '极度'], ['欠', '适度']),
			(['无胆色大便'], ['大便', '稀便']),
		]


	def get_ant_enhance_pairs(self):
		term_to_antonyms = {}
		antonyms_pairs = self.get_antonyms_pairs()
		for terms1, terms2 in antonyms_pairs:
			for t in terms1:
				term_to_antonyms[t] = terms2
			for t in terms2:
				term_to_antonyms[t] = terms1
		t_sorted_by_len = sorted(list(term_to_antonyms.keys()), key=lambda item:len(item), reverse=True)
		hpo_terms = self.get_cns_list()
		neg_pairs = []
		for term, ants in term_to_antonyms.items():
			neg_pairs.extend([(term, a) for a in ants])
		for hpo_term in hpo_terms:
			for t in t_sorted_by_len:
				idx = hpo_term.find(t)
				if idx != -1: # found
					prefix, post_fix = hpo_term[:idx], hpo_term[idx+len(t):]
					neg_pairs.extend([(hpo_term, prefix+at+post_fix) for at in term_to_antonyms[t]])
					# neg_pairs.append((hpo_term, prefix+post_fix))
					break
		neg_pairs = unique_pairs(neg_pairs)
		return neg_pairs


	def get_neg_detect_enhance_pairs(self):
		neg_prefix_list = [
			'未引出', '未发现', '未见', '没有', '否认', '无', '非',
			'不明显', '未再', '未出现', '不符合', '不考虑', '除外', '未诉', '不伴', '未闻及'
		]
		neg_postfix_list = ['未引出', '未见', '未出现', '不明显', '不考虑', '除外', '（-）', '(-)', '阴性']
		neg_pairs = []
		hpo_terms = self.get_cns_list()
		for hpo_term in hpo_terms:
			neg_pairs.extend([(hpo_term, neg_prefix + hpo_term) for neg_prefix in neg_prefix_list])
			neg_pairs.extend([(hpo_term, hpo_term + neg_postfix) for neg_postfix in neg_postfix_list])
		return unique_pairs(neg_pairs)


	def get_parent_child_pairs(self):
		ret_pairs = []
		hpo_dict = self.get_hpo_dict()
		for hpo, info in hpo_dict.items():
			ret_pairs.extend([(hpo, child_hpo) for child_hpo in info.get('CHILD', [])])
		return ret_pairs


	def get_brother_pairs(self):
		ret_pairs = []
		hpo_dict = self.get_hpo_dict()
		for hpo, info in hpo_dict.items():
			ret_pairs.extend(list(itertools.combinations(info.get('CHILD', []), 2)))
		return ret_pairs


class SynDictReader(object):
	def __init__(self):

		self.SAVE_FOLDER = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/preprocess/knowledge/UMLS'

		self.HPO_TO_SOURCE_SYN_TERMS_JSON = os.path.join(self.SAVE_FOLDER, 'HPOToSourceSynTerms.json')
		self.hpo_to_source_syn_terms = None
		self.HPO_TO_BG_EVALUATE_SYN_TERMS_JSON = os.path.join(self.SAVE_FOLDER, 'HPOToBGEvaSynTerms.json')
		self.hpo_to_bg_eva_syn_terms = None

		self.CUI_TO_SOURCE_SYN_TERMS_JSON = os.path.join(self.SAVE_FOLDER, 'CUIToSourceSynTerms.json')
		self.cui_to_source_syn_terms = None
		self.CUI_TO_BG_EVALUATE_SYN_TERMS_JSON = os.path.join(self.SAVE_FOLDER, 'CUIToBGEvaSynTerms.json')
		self.cui_to_bg_eva_syn_terms = None

		self.CUI_TO_DICT_SYN_TERMS_JSON = os.path.join(self.SAVE_FOLDER, 'CuiToDictSynTerms.json')
		self.cui_to_dict_syn_terms = None

		self.del_words = get_del_words()


	def get_hpo_to_codestrs(self):
		"""
		Returns:
			dict: {hpo: [code_str1, ...]}
		"""

		def get_code_strs(hpo):
			return [hpo, hpo.lower(), hpo.split(':').pop()]

		hpo_reader = HPOReader()
		hpo_codes = list(hpo_reader.get_hpo_dict().keys())
		ret_dict = {}
		for hpo in hpo_codes:
			code_strs = get_code_strs(hpo)
			ret_dict[hpo] = code_strs
		old2new = hpo_reader.get_old_map_new_hpo_dict()
		for old_hpo, new_hpo in old2new.items():
			ret_dict[new_hpo].extend(get_code_strs(old_hpo))
		return ret_dict


	def process_hpo_to_syns(self, hpo_to_syns):
		ret = {}
		for hpo, syns in list(hpo_to_syns.items()):
			syns = [t for t in syns if len(t) > 1 and t not in self.del_words]
			if len(syns) > 0:
				ret[hpo] = syns
		return ret


	@check_load('hpo_to_source_syn_terms', 'HPO_TO_SOURCE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_source_syn_terms_(self):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		assert False


	def get_hpo_to_source_syn_terms(self):
		return self.process_hpo_to_syns(self.get_hpo_to_source_syn_terms_())


	def get_hpo_to_dict_syn_terms(self):
		hpo_to_cui = UMLSReader().get_hpo_to_cui()
		cui_to_dict_syn_terms = self.get_cui_to_dict_syn_terms()
		hpo_to_syns = {hpo: cui_to_dict_syn_terms[cui] for hpo, cui in hpo_to_cui.items() if cui in cui_to_dict_syn_terms}
		return self.process_hpo_to_syns(hpo_to_syns)


	def get_hpo_to_source_dict_syn_terms(self):
		return self.combine_code_to_syn_terms(
			self.get_hpo_to_source_syn_terms(),
			self.get_hpo_to_dict_syn_terms())


	@check_load('hpo_to_bg_eva_syn_terms', 'HPO_TO_BG_EVALUATE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_hpo_to_syn_terms_with_bg_evaluate_(self):
		"""
		Returns:
			dict: {hpo_code: [cns_term, ...]}
		"""
		assert False


	def get_hpo_to_syn_terms_with_bg_evaluate(self):
		return self.process_hpo_to_syns(self.get_hpo_to_syn_terms_with_bg_evaluate_())


	def get_hpo_to_source_bg_dict_syn_terms(self):
		return self.combine_code_to_syn_terms(
			self.get_hpo_to_syn_terms_with_bg_evaluate(),
			self.get_hpo_to_dict_syn_terms())


	@check_load('cui_to_source_syn_terms', 'CUI_TO_SOURCE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_cui_to_source_syn_terms(self):
		assert False


	@check_load('cui_to_bg_eva_syn_terms', 'CUI_TO_BG_EVALUATE_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_cui_to_syn_terms_with_bg_evaluate(self):
		assert False


	@check_load_save('cui_to_dict_syn_terms', 'CUI_TO_DICT_SYN_TERMS_JSON', JSON_FILE_FORMAT)
	def get_cui_to_dict_syn_terms(self):
		"""FIXME: 解决过长
		Returns:
			dict: {cui_code: [term1, term2, ...]}
		"""
		def add_eng_to_cns_terms(eng_to_cns, info_dicts):
			for info in info_dicts:
				for eng_term in info['ENG_NAMES']:
					if 'CNS_NAMES' not in info:
						print('No CNS_NAMES provided, ignore: {}'.format(info))
						continue
					dict_set_update(eng_term, [cns_name.strip() for cns_name in info['CNS_NAMES'] if cns_name.strip() and contain_cns(cns_name)], eng_to_cns)

		print('reading dicts...')
		eng_to_cns = {}
		add_eng_to_cns_terms(eng_to_cns, MedDict12Reader().get_info_dicts())
		add_eng_to_cns_terms(eng_to_cns, BoarlandsReader().get_info_dicts())
		add_eng_to_cns_terms(eng_to_cns, NewGeneralMedDict().get_info_dicts())
		add_eng_to_cns_terms(eng_to_cns, XiangyaMedDict().get_info_dicts())
		add_eng_to_cns_terms(eng_to_cns, EngCnsMedDict().get_info_dicts())
		print('reading dicts...done')

		print('processing eng_to_cns...', end='')
		# 删除缩写
		for eng in list(eng_to_cns.keys()):
			cns_set = eng_to_cns[eng]
			for cns in cns_set:
				if re.search('\[=.*\]', cns):
					del eng_to_cns[eng]
					break

		# 大写转小写
		lower_eng_to_cns = {}
		for eng, cns_set in eng_to_cns.items():
			dict_set_update(eng.lower(), cns_set, lower_eng_to_cns)
		eng_to_cns = lower_eng_to_cns
		print('done')

		print('reading umls...', end='')
		umls_reader = UMLSReader()
		eng_terms = umls_reader.get_eng_list()
		aui_lists = umls_reader.get_aui_lists()
		aui_to_cui = umls_reader.get_aui_to_cui()
		print('done')

		print('generating cui_to_syn...', end='')
		cui_to_syn, match_eng_set = {}, set()
		for eng, aui_list in tqdm(zip(eng_terms, aui_lists)):
			cui_set = {aui_to_cui[aui] for aui in aui_list}
			if len(cui_set) > 1:
				continue
			cui = list(cui_set)[0]

			eng = eng.lower()
			if eng in eng_to_cns:
				match_eng_set.add(eng)
				dict_set_update(cui, eng_to_cns[eng], cui_to_syn)

		id = 1
		for eng, cns_set in eng_to_cns.items():
			if eng not in match_eng_set and len(cns_set) > 1:
				my_cui = 'M{:07}'.format(id)
				cui_to_syn[my_cui] = cns_set

		for cui in cui_to_syn:
			cui_to_syn[cui] = list(cui_to_syn[cui])
		print('done')
		return cui_to_syn


	def combine_code_to_syn_terms(self, code_to_syns1, code_to_syns2):
		code_to_syns1 = deepcopy(code_to_syns1)
		for code, syns in code_to_syns2.items():
			if code in code_to_syns1:
				code_to_syns1[code] = list(set(code_to_syns1[code] + syns))
			else:
				code_to_syns1[code] = syns
		return code_to_syns1


	def get_cui_to_source_dict_syn_terms(self):
		return self.combine_code_to_syn_terms(
			self.get_cui_to_source_syn_terms(),
			self.get_cui_to_dict_syn_terms())


	def get_cui_to_source_bg_dict_syn_terms(self):
		return self.combine_code_to_syn_terms(
			self.get_cui_to_syn_terms_with_bg_evaluate(),
			self.get_cui_to_dict_syn_terms())


	def get_hpo_neg_with_jaccard(self, min_sim, hpo_to_syns=None, neg_from_hpo=True, interval=500, cpu_use=12, select_from_sim=None, mark=''):
		"""
		source_bg_dict,neg_from_hpo=True: 0.2: 3931659; 0.3: 893402; 0.4: 276533; 0.5: 103464; 0.6: 29248; 0.7: 9146
		source_dict,neg_from_hpo=True: 0.2, 2655072; 0.3, 641311; 0.4, 205697; 0.5, 80182; 0.6, 24084; 0.7, 7829
		source,neg_from_hpo=True: 0.2, 2013548; 0.3, 514867; 0.4, 170281; 0.5, 68052; 0.6, 21243; 0.7, 7061
		Returns:
			list: [(text_a, text_b, sim), ...]
		"""
		def get_save_pkl(min_sim, pos_hpo):
			return os.path.join(self.SAVE_FOLDER, f'hpo_{mark}_neg-neg_from_hpo{pos_hpo}-min_sim{min_sim}.pkl')

		save_pkl = get_save_pkl(min_sim, neg_from_hpo)
		if os.path.exists(save_pkl):
			return get_load_func(PKL_FILE_FORMAT)(save_pkl)

		if select_from_sim is not None:
			from_pkl = get_save_pkl(select_from_sim, neg_from_hpo)
			assert os.path.exists(from_pkl)
			neg_samples = get_load_func(PKL_FILE_FORMAT)(from_pkl)
			neg_samples = [ns for ns in neg_samples if ns[2] >= min_sim]
		else:
			hpo_reader = HPOReader()
			hpo_dict = hpo_reader.get_hpo_dict()
			hpo_to_cns = hpo_reader.get_hpo_to_cns()
			hpo_to_ances_set = {hpo: get_all_ancestors(hpo, hpo_dict) for hpo in hpo_dict}
			neg_samples, cand_pairs = [], []  # (text_a, text_b, sim)
			for i, (hpo, syns) in tqdm(enumerate(hpo_to_syns.items(), 1)):
				for tgt_hpo in hpo_to_syns:
					if tgt_hpo in hpo_to_ances_set[hpo]:
						continue
					if hpo in hpo_to_ances_set[tgt_hpo]:
						continue
					tgt_syns = hpo_to_syns[tgt_hpo]
					src_syns = [hpo_to_cns.get(hpo, '')] if neg_from_hpo else syns
					if not src_syns:
						continue
					cand_pairs.extend(list(itertools.product(src_syns, tgt_syns)))
				if i % interval == 0 or i == len(hpo_to_syns):
					sim_list = cal_jaccard_sim_list(cand_pairs, cpu_use=cpu_use, chunk_size=50000)
					neg_samples.extend([(pair[0], pair[1], sim) for pair, sim in zip(cand_pairs, sim_list) if sim >= min_sim])
					cand_pairs = []
					print('{}/{} ({:.4}%); Length of neg_samples: {}\n {}'.format(
						i, len(hpo_to_syns), i*100./len(hpo_to_syns), len(neg_samples),
						random.sample(neg_samples, min(10, len(neg_samples))) if neg_samples else []))
			neg_samples = unique_lists(neg_samples, lambda items: tuple(sorted(items[:2])))
		print(f'Min Sim = {min_sim}; Total pairs = {len(neg_samples)}')
		json.dump(random.sample(neg_samples, min(1000, len(neg_samples))), open(os.path.splitext(save_pkl)[0]+'-small.json', 'w'), indent=2, ensure_ascii=False)
		get_save_func(PKL_FILE_FORMAT)(neg_samples, save_pkl)
		return neg_samples


class ChipReader(object):
	def __init__(self):
		CHIP_FOLDER = os.path.join(DATA_PATH, 'raw', 'chip')
		self.RAW_TRAIN_XLSX = os.path.join(CHIP_FOLDER, 'train_chip.xlsx')
		self.RAW_VAL_XLSX = os.path.join(CHIP_FOLDER, 'val_chip.xlsx')
		self.RAW_TEST_XLSX = os.path.join(CHIP_FOLDER, 'test_chip.xlsx')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'chip')
		self.ALL_PAIRS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'all_pairs.json')
		self.all_pairs = None


	def read_xlsx(self, path):
		"""
		Returns:
			list: [(text_a, text_b), ...]
		"""
		ret_pairs = []
		df = pd.read_excel(path)
		for s1, s2 in df.values:
			if s1.find('+') != -1 or s2.find('##') != -1:
				continue
			ret_pairs.append((s1.strip(), s2.strip()))
		return ret_pairs


	@check_load_save('all_pairs', 'ALL_PAIRS_JSON', JSON_FILE_FORMAT)
	def get_all_pairs(self):
		"""
		Returns:
			list: [(raw_term, std_term), ...]
		"""
		return self.read_xlsx(self.RAW_TRAIN_XLSX) + self.read_xlsx(self.RAW_VAL_XLSX) + self.read_xlsx(self.RAW_TEST_XLSX)


class PUMCReader(object):
	def __init__(self):

		self.RAW_DATA_FOLDER = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/core/data/raw/PUMC'
		self.RAW_PATIENT_FOLDER = os.path.join(self.RAW_DATA_FOLDER, 'case87-doc-hy-strict-enhance')
		self.TEST_PID_JSON = os.path.join(self.RAW_DATA_FOLDER, 'val_test_split', 'test_pids.json')
		self.VAL_PID_JSON = os.path.join(self.RAW_DATA_FOLDER, 'val_test_split', 'val_pids.json')


	def get_json_pairs(self, json_path):
		"""
		Returns:
			list: [(ehr_term, hpo_term), ...]
		"""
		ret_pairs = []
		field2info = json.load(open(json_path))
		for field_info in field2info.values():
			text = field_info['RAW_TEXT']
			for entity_dict in field_info['ENTITY_LIST']:
				text_a = ''.join([text[s: e] for s, e in entity_dict['SPAN_LIST']])
				text_b = entity_dict['HPO_TEXT']
				ret_pairs.append((text_a, text_b))
		return ret_pairs


	def get_doc_tag_pairs(self, data_type='test'):
		"""
		Args:
			data_type (str): 'eval' | 'test'
		Returns:
			list: [(ehr_term, hpo_term), ...]
		"""
		assert data_type == 'test' or data_type == 'eval'
		ret_pairs = []
		pids = json.load(open(self.TEST_PID_JSON if data_type == 'test' else self.VAL_PID_JSON))
		for pid in pids:
			ret_pairs.extend(self.get_json_pairs(os.path.join(self.RAW_PATIENT_FOLDER, f'{pid}.json')))
		return list(set(ret_pairs))


	def get_pids(self, data_type):
		if data_type == TEST_DATA:
			return json.load(open(self.TEST_PID_JSON))
		elif data_type == VALIDATION_DATA:
			return json.load(open(self.VAL_PID_JSON))
		else:
			raise RuntimeError('Unknown data type: {}'.format(data_type))


	def get_json_paths(self, data_type):
		pids = self.get_pids(data_type)
		return [os.path.join(self.RAW_PATIENT_FOLDER, f'{pid}.json') for pid in pids]


class MedDict12Reader(object):
	def __init__(self):
		"""Note: convert .mobi to .txt using convertio (https://convertio.co/zh/mobi-converter)
		"""
		self.RAW_MED_DICT_TXT = os.path.join(DATA_PATH, 'raw', 'med_dict_mobi', '[5_04]12合1医学类词典.txt')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'med_dict_12')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.INFO_DICTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'info_dicts.json')
		self.info_dicts = None


	@check_load_save('info_dicts', 'INFO_DICTS_JSON', JSON_FILE_FORMAT)
	def get_info_dicts(self):
		"""
		Returns:
			list: [{'ENG_NAMES': str, 'CNS_NAMES': str, 'SOURCE': str}]
		"""
		lines = open(self.RAW_MED_DICT_TXT).read().splitlines()
		lines = [line for line in lines if line.strip()]
		source_idices = [i for i, line in enumerate(lines) if line.startswith('《')]
		info_dicts = []
		for source_idx in source_idices:
			source = re.search('《(.*)》', lines[source_idx]).group(1).strip()
			cns_name = lines[source_idx+1].replace('"', '').strip()
			eng_name = lines[source_idx+2].replace('"', '').strip()
			if contain_cns(eng_name):
				cns_name, eng_name = eng_name, cns_name
			info_dicts.append({
				'SOURCE': source,
				'CNS_NAMES': [cns_name],
				'ENG_NAMES': [eng_name]
			})
		return info_dicts


class BoarlandsReader(object):
	def __init__(self):
		self.RAW_MED_DICT_MDX = os.path.join(DATA_PATH, 'raw', 'boarlands', '英中医学辞海2009', '英中医学辞海2009.mdx')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'boarlands')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.INFO_DICTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'info_dicts.json')
		self.info_dicts = None


	@check_load_save('info_dicts', 'INFO_DICTS_JSON', JSON_FILE_FORMAT)
	def get_info_dicts(self):
		"""
		Returns:
			list: [{'ENG_NAMES': str, 'CNS_NAMES': str, 'EQUAL_TO': str, 'RELATED_TO': str}]; RELATED_TO指"习惯用语"条目下的词
		"""
		def process_cns_str(cns_str):
			cns_str = re.sub('[①②③④⑤⑥⑦⑧⑨]', '', cns_str)
			info = {}
			cns_items = re.split('[:：]', cns_str, maxsplit=1)
			if len(cns_items) > 2:
				print('cns_str:', cns_str)
				print('cns_items:', len(cns_items) , cns_items)
			assert len(cns_items) <= 2
			cns_name = cns_items[0].strip()
			cns_name = re.sub('[［\[]', '〔', cns_name)
			cns_name = re.sub('[\]］]', '〕', cns_name)
			cns_name = re.sub('(〔|^〔?)(.*\d+~\d+.*|单.*|复.*|.*国.*|.*医师.*|.*学家.*|构词成分|NA|USP|拉.*|希.*|德.*|法.*|印第安|EC.*)〕', '', cns_name)
			cns_name = cns_name.replace('〔', '（').replace('〕', '）').strip()
			info['CNS_NAMES'] = re.split('[,，;；]', cns_name)
			if len(cns_items) == 2:
				info['CNS_DEF'] = cns_items[1].strip()
				search_obj = re.search('^同(.*?)；?$', info['CNS_DEF'])
				if search_obj:
					info['EQUAL_TO'] = search_obj.group(1).strip()
			return info

		from bert_syn.mdx_pkg.readmdict import MDX, MDD
		mdx = MDX(self.RAW_MED_DICT_MDX)
		info_dicts = []
		items = mdx.items()
		for item in tqdm(items):
			raw_eng_name = item[0].decode('utf-8')
			info = {'ENG_NAMES': [eng_name.strip() for eng_name in re.split('[；;]', raw_eng_name)]}
			raw_cns_str = item[1].decode('utf-8')
			search_obj = re.search(r'<img src="file://dot.gif">(.*?)<br>', raw_cns_str)
			if not search_obj:
				info_dicts.append(info)
				continue
			cns_str = search_obj.group(1)
			for sub_cns_str in re.split('[;；][[①②③④⑤⑥⑦⑧⑨]', cns_str):
				info_dicts.append({**info, **process_cns_str(sub_cns_str)})
			related_term_idx = raw_cns_str.find('<font color="DarkBlue">习惯用语</font>')
			if related_term_idx != -1:
				it = re.finditer('<font size="+1">&nbsp&nbsp<b>(.*?)</b></font><br><img src="file://dot.gif">(.*?)<br>', raw_cns_str)
				for match_obj in it:
					raw_rel_eng_name = match_obj.group(1)
					rel_info = {
						'ENG_NAMES': [rel_eng_name.strip() for rel_eng_name in re.split('[；;]', raw_rel_eng_name)],
						'RELATED_TO': info['ENG_NAME']
					}
					cns_str = match_obj.group(2).strip()
					for sub_cns_str in re.split('[;；][①②③④⑤⑥⑦⑧⑨]', cns_str):
						info_dicts.append({**rel_info, **process_cns_str(sub_cns_str)})
		return info_dicts


class NewGeneralMedDict(object):
	def __init__(self):
		"""Note: convert .mobi to .txt using convertio (https://convertio.co/zh/mobi-converter); Actually the same as xiang_ya
		"""
		self.RAW_MED_DICT_MDX = os.path.join(DATA_PATH, 'raw', 'med_dict_mobi', '5_06_新编全医学大字典.txt')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'new_general_med_dict')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.INFO_DICTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'info_dicts.json')
		self.info_dicts = None


	@check_load_save('info_dicts', 'INFO_DICTS_JSON', JSON_FILE_FORMAT)
	def get_info_dicts(self):
		"""
		Returns:
			list: [{'ENG_NAMES': str, 'CNS_NAMES': str}]
		"""
		content = open(self.RAW_MED_DICT_MDX).read()
		it = re.finditer('\n`1`(.*?)`2`(.*?)\n', content)
		info_dicts = []
		for match_obj in tqdm(it):
			cns_name = match_obj.group(1).strip()
			eng_name = match_obj.group(2).strip()
			if contain_cns(eng_name) and not contain_cns(cns_name):
				cns_name, eng_name = eng_name, cns_name
			if contain_cns(eng_name):
				print('cns_name = {} | eng_name = {} | obj = {}'.format(cns_name, eng_name, match_obj.group()))
			eng_names = re.split('[;；]', eng_name)
			if cns_name.find('-') != -1: # found '-'
				cns_names = [cns_name]
			else:
				cns_names = re.split('[,，;；]', cns_name)
			info_dicts.append({'CNS_NAMES': cns_names, 'ENG_NAMES': eng_names})
		return info_dicts


class XiangyaMedDict(object):
	def __init__(self):
		"""Note: convert .mobi to .html using KindleUnpack (https://github.com/kevinhendricks/KindleUnpack)
		"""
		self.RAW_MED_DICT_HTML = os.path.join(DATA_PATH, 'raw', 'med_dict_mobi', '[5_05]湘雅医学专业英汉词典.html')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'xiang_ya_med_dict')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.INFO_DICTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'info_dicts.json')
		self.info_dicts = None


	@check_load_save('info_dicts', 'INFO_DICTS_JSON', JSON_FILE_FORMAT)
	def get_info_dicts(self):
		"""
		Returns:
			list: [{'ENG_NAMES': [str], 'CNS_NAMES': [str]}]
		"""
		content = open(self.RAW_MED_DICT_HTML).read()
		it = re.finditer(r'<idx:entry .*?><idx:orth value="(.*?)"></idx:orth><h2>(.*?)</h2> <br><font color="#0000ff">(.*?)<br/> </font> </idx:entry>', content)
		info_dicts = []
		for match_obj in tqdm(it):
			info = {}
			raw_eng_name1, raw_eng_name, raw_cns_name = match_obj.group(1).strip(), match_obj.group(2).strip(), match_obj.group(3).strip()
			if raw_eng_name1 != raw_eng_name:
				print('Confilict: ', raw_eng_name1, ' | ', raw_eng_name)
			if contain_cns(raw_eng_name):
				raw_cns_name, raw_eng_name = raw_eng_name, raw_cns_name
			eng_names = [s.strip() for s in re.split(';|；|<br>', raw_eng_name)]

			info['ENG_NAMES'] = eng_names
			cns_names = re.split('<br>', raw_cns_name)
			if len(cns_names) == 1:
				cns_name = cns_names[0]
				if cns_name.find('-') != -1 or re.search('\[=.*\]', cns_name):
					info_dicts.append({**info, **{'CNS_NAMES':[cns_name.strip()]}})
				else:
					info_dicts.append({**info, **{'CNS_NAMES':[s.strip() for s in re.split('[,，;；]', cns_name)]}})
			else:
				info_dicts.append({**info, **{'CNS_NAMES':[s.strip() for s in cns_names]}})
		return info_dicts


class EngCnsMedDict(object):
	def __init__(self):
		"""Note: convert .mobi to .html using KindleUnpack (https://github.com/kevinhendricks/KindleUnpack)
		"""
		self.RAW_MED_DICT_HTML = os.path.join(DATA_PATH, 'raw', 'med_dict_mobi', '[5_08]英汉汉英医学辞典.html')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'eng_cns_med_dict')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.INFO_DICTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'info_dicts.json')
		self.info_dicts = None


	@check_load_save('info_dicts', 'INFO_DICTS_JSON', JSON_FILE_FORMAT)
	def get_info_dicts(self):
		"""
		Returns:
			list: [{'ENG_NAMES': [str], 'CNS_NAMES': [str], 'SOURCE': str}]
		"""
		content = open(self.RAW_MED_DICT_HTML).read()
		it = re.finditer('<idx:entry scriptable="yes"><idx:orth value=".*?"></idx:orth><h2>.*?</h2> <br><font color=".*?" size="5"><b>(.*?)</font><br/> <p><font color=".*?" size="5"><b>(.*?)</font><br/> <br>(.*?)<br/> </idx:entry>', content)
		info_dicts = []
		for match_obj in tqdm(it):
			source = match_obj.group(1).strip()
			cns_name, eng_name = match_obj.group(2), match_obj.group(3)
			if source == '《英汉医学词典》':
				cns_name, eng_name = eng_name, cns_name

			search_obj = re.search('(〖英文名〗|\(拉\)|\[生词求解\])(.*?)(<br/>|$)', eng_name)
			if search_obj:
				eng_name = search_obj.group(2)
			eng_name = re.sub('(\[单数?\]|\[复数?\])', '', eng_name)
			eng_names = re.split('[;；]', eng_name)

			search_obj = re.search('〖中文名〗(.*?)<br/>', cns_name)
			if search_obj:
				cns_name = search_obj.group(1)
			search_obj = re.search('^(.*?)<br/>', cns_name)
			if search_obj:
				cns_name = search_obj.group(1)
			if cns_name.find('-') != -1:
				cns_names = [cns_name]
			else:
				cns_names = re.split('[,，;；]', cns_name)


			info_dicts.append({
				'SOURCE': source,
				'ENG_NAMES': eng_names,
				'CNS_NAMES': cns_names
			})
		return info_dicts


class HitReader(object):
	def __init__(self):
		self.RAW_MED_DICT_TXT = os.path.join(DATA_PATH, 'raw', 'hit', '同义词库.txt')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'hit')
		self.CODE_TO_SYNS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'code_to_syns.json')
		self.code_to_syns = None


	@check_load_save('code_to_syns', 'CODE_TO_SYNS_JSON', JSON_FILE_FORMAT)
	def get_code_to_syns(self):
		"""
		Returns:
			dict: {code: [syn_term1, syn_term2, ...]}
		"""
		lines = open(self.RAW_MED_DICT_TXT).read().splitlines()
		lines = [line for line in lines if line.strip()]
		code_to_syns = {}
		for line in lines:
			items = line.split(' ')
			code = items[0][:-1]
			mark = items[0][-1]
			if mark == '=':
				code_to_syns[code] = items[1:]
		return code_to_syns


class AntonymReader(object):
	def __init__(self):
		self.RAW_DICT_TXT = os.path.join(DATA_PATH, 'raw', 'antonym', '反义词库.txt')
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'antonym')
		self.TERM_PAIRS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'term_pairs.json')
		self.term_pairs = None

	@check_load_save('term_pairs', 'TERM_PAIRS_JSON', JSON_FILE_FORMAT)
	def get_term_pairs(self):
		lines = open(self.RAW_DICT_TXT).read().strip().splitlines()
		ret_pairs = []
		for pair_str in lines:
			pair_str = re.sub('──|——|―|—|--', '|', pair_str)
			pair_str = re.sub('\|+', '|', pair_str)
			pair = pair_str.split('|')
			if len(pair) != 2:
				print(pair_str, pair)
			assert len(pair) == 2
			ret_pairs.append((pair[0].strip(), pair[1].strip()))
		return ret_pairs


class SCELReader(object):
	def __init__(self):
		self.SAVE_FOLDER = os.path.join(DATA_PATH, 'raw', 'sougou')


	def get_vocab(self, file_name):
		terms = open(os.path.join(self.SAVE_FOLDER, f'{file_name}.txt', )).read().strip().splitlines()
		# if '白细胞减少' in terms:
		# 	print(file_name)
		# 	assert False
		return list(set([t.strip() for t in terms if len(t.strip()) > 0]))


	def get_multi_vocab(self, file_names):
		ret_vocab = set()
		for file_name in file_names:
			ret_vocab.update(self.get_vocab(file_name))
		return list(ret_vocab)


class StopwordsReader(object):
	def __init__(self):
		pass


	def get_stop_words(self):
		return [
			'诊断', '诊断为', '既往史', '个人史', '提示', '就诊', '病程中', '发现', '未诊治', '症状', '家族史', '病史', '同时', '目前',
			'此后', '具体不详', '患者', '婚育史'
		]


def gen_context_pos_pairs(words, corpus, sample_num=None, matcher='exact'):
	"""
	Args:
		words (iterable): [str1, str2, ...]
		corpus (list): [sentence1, sentence2, ...]
		sample_num (int):
		matcher (str): 'exact' | 'bag'
	Returns:
		list: [(text_a, text_b), ...]
	"""
	from bert_syn.core.span_generator import ExactTermMatcher, BagTermMatcher, MaxInvTextSearcher
	word2codes = {w: [i] for i, w in enumerate(words)}
	matcher = ExactTermMatcher(word2codes=word2codes) if matcher == 'exact' else BagTermMatcher(word2codes=word2codes)
	searcher = MaxInvTextSearcher(matcher)
	all_pos_pairs = set()
	for sent in corpus:
		sent = sent.strip()
		wids, pos_list = searcher.search(sent)
		if len(wids) > 0:
			all_pos_pairs.update([(words[wid], sent) for wid in wids if words[wid] != sent])
	all_pos_pairs = list(all_pos_pairs)
	if sample_num is not None:
		all_pos_pairs = random.sample(all_pos_pairs, min(len(all_pos_pairs), sample_num))
	return all_pos_pairs


def gen_random_ctx_pos_pairs(words, corpus, sample_num, matcher='exact'):
	def count_list_to_probs(ll):
		ll = np.array(ll, dtype=np.float64)
		return ll / ll.sum()
	from bert_syn.core.span_generator import ExactTermMatcher, BagTermMatcher, MaxInvTextSearcher
	word2codes = {w:[i] for i, w in enumerate(words)}
	matcher = ExactTermMatcher(word2codes=word2codes) if matcher == 'exact' else BagTermMatcher(word2codes=word2codes)
	searcher = MaxInvTextSearcher(matcher)
	ctx_char_dist, pre_len_dist, post_len_dist = Counter(), Counter(), Counter()
	for sent in corpus:
		sent = sent.strip()
		wids, pos_list = searcher.search(sent)
		if len(wids) > 0:
			ctx_char_dist.update(words[wids[0]])
			pre_len = pos_list[0][0]
			post_len = len(sent) - pos_list[0][1]
			pre_len_dist[pre_len] += 1
			post_len_dist[post_len] += 1
	chars, char_probs = zip(*ctx_char_dist.items())
	char_probs = count_list_to_probs(char_probs)

	pre_lens, pre_len_probs = zip(*pre_len_dist.items())
	pre_len_probs = count_list_to_probs(pre_len_probs)

	post_lens, post_len_probs = zip(*post_len_dist.items())
	post_len_probs = count_list_to_probs(post_len_probs)

	pre_sample_lens = np.random.choice(pre_lens, sample_num, p=pre_len_probs)
	post_sample_lens = np.random.choice(post_lens, sample_num, p=post_len_probs)
	center_words = np.random.choice(words, sample_num)
	all_pos_pairs = set()
	for pre_len, post_len, center_w in tqdm(zip(pre_sample_lens, post_sample_lens, center_words)):
		prefix = ''.join(np.random.choice(chars, pre_len, p=char_probs))
		postfix = ''.join(np.random.choice(chars, post_len, p=char_probs))
		all_pos_pairs.add((center_w, prefix+center_w+postfix))
	return all_pos_pairs


def get_anatomy_neg_pairs():
	anatomy_words = set(SCELReader().get_multi_vocab([
		'人体解剖学名词（中文）2009', '人体组织名称', '解剖学词库', '人体解剖学名词【官方推荐】',
		'细胞生物学词库', '微生物学常用词汇',
	]))
	anatomy_words.remove('异常')
	hpo2syns = SynDictReader().get_hpo_to_source_syn_terms()
	symptom_terms = set(t for syns in hpo2syns.values() for t in syns)
	neg_pairs = []
	for symptom in symptom_terms:
		for w in anatomy_words:
			if symptom.find(w) > -1:
				neg_pairs.append((w, symptom))
	return neg_pairs


def get_symptom_combine_pos_pairs(sample_num):
	sample_num = sample_num // 4
	hpo2cns = HPOReader().get_hpo_to_cns()
	hpo2syns = SynDictReader().get_hpo_to_source_syn_terms()
	all_hpos = list(hpo2syns.keys())
	pre_hpos = random.sample(all_hpos, sample_num)
	post_hpos = random.sample(all_hpos, sample_num)
	all_pos_pairs = set()
	for i, (pre_hpo, post_hpo) in enumerate(zip(pre_hpos, post_hpos)):
		if i % 2 == 1:
			pre_hpo, post_hpo = post_hpo, pre_hpo
		if pre_hpo not in hpo2cns or post_hpo not in hpo2syns:
			continue
		term1, term2 = hpo2cns[pre_hpo], random.choice(hpo2syns[post_hpo])
		all_pos_pairs.update([
			(term1, term1 + term2), (term1, term2+term1),
			(term2, term1 + term2), (term2, term2 + term1),
		])
	return all_pos_pairs


def get_cui_select_pos_pairs(cui_to_syns, sample_num):
	keep_tuis = [
		'T047', # Disease or Syndrome; 43536
		'T184', # Sign or Symptom; 2584
		'T048', # Mental or Behavioral Dysfunction (精神或行为障碍); 3237
		'T191', # Neoplastic Process (肿瘤过程); 6301
		'T049', # Cell or Molecular Dysfunction (细胞或分子功能障碍); 352
		'T050', # Experimental Model of Disease; 25
		# 'T046' # Pathologic Function; 9075; del
		# 'T037', # Injury or Poisoning; 66885; del
		'T034', # Laboratory or Test Result; 454
		'T033', # Finding; 13585
	]
	keep_cuis = UMLSReader().get_cuis_given_tuis(keep_tuis)
	cui_to_syns = {cui: list(set(cui_to_syns[cui])) for cui in keep_cuis if cui in cui_to_syns}
	keep_cuis = [cui for cui, syns in cui_to_syns.items() if len(syns) > 1]
	print('Length of keep cui: ', len(keep_cuis))
	sample_cuis = np.random.choice(keep_cuis, sample_num)

	return [tuple(random.sample(cui_to_syns[cui], 2)) for cui in sample_cuis]


def get_hpo_def_pos_pairs():
	chpo_dict = HPOReader().get_chpo_dict()
	ret_pairs = []
	for hpo, info in chpo_dict.items():
		if info.get('CNS_NAME', '') and info.get('CNS_DEF', ''):
			ret_pairs.append((info['CNS_NAME'], info['CNS_DEF']))
	return ret_pairs


def get_hpo_eng_pos_pairs():
	hpo_to_cns = HPOReader().get_hpo_to_cns()
	hpo_dict = HPOReader().get_hpo_dict()
	ret_pairs = []
	for hpo, cns in hpo_to_cns.items():
		ret_pairs.append((cns, hpo_dict[hpo]['ENG_NAME']))
	return ret_pairs


class DataHelper(object):
	def __init__(self):
		self.chip_reader = None
		self.hpo_reader = None
		self.umls_reader = None
		self.pumc_reader = None
		self.syn_dict_reader = None
		self.hit_reader = None
		self.antonym_reader = None
		self.NEG_LABEL = '0'
		self.POS_LABEL = '1'
		self.MID_LABEL = '0.5'
		self.PSEUDO_LABEL = self.NEG_LABEL


	@check_return('chip_reader')
	def get_chip_reader(self):
		return ChipReader()

	@check_return('hpo_reader')
	def get_hpo_reader(self):
		return HPOReader()

	@check_return('pumc_reader')
	def get_pumc_reader(self):
		return PUMCReader()

	@check_return('syn_dict_reader')
	def get_syn_dict_reader(self):
		return SynDictReader()

	@check_return('umls_reader')
	def get_umls_reader(self):
		return UMLSReader()

	@check_return('hit_reader')
	def get_hit_reader(self):
		return HitReader()

	@check_return('antonym_reader')
	def get_antonym_reader(self):
		return AntonymReader()

	def get_samples_info(self, samples):
		info_dict = {'SAMPLE_NUM': len(samples)}
		label_counter = Counter([label for text_a, text_b, label in samples])
		info_dict['LABEL_COUNT'] = label_counter.most_common()
		length_counter = Counter()
		for text_a, text_b, label in samples:
			length_counter[len(text_a)] += 1
			length_counter[len(text_b)] += 1
		info_dict['STR_LEN_COUNT'] = length_counter.most_common()
		return info_dict


	def max_seq_length_filter(self, pairs, max_seq_len=None):
		return pairs if max_seq_len is None else [p for p in pairs if (len(p[0]) + len(p[1])) <= max_seq_len-3]


	@timer
	def split_and_save_samples(self, samples, save_folder, train_size=0.9, swap_aug=True):
		"""
		Args:
			samples (list): [(text_a, text_b, label), ...]
		"""
		has_eval = (train_size != 1.0)
		if has_eval:
			train_samples, eval_samples = train_test_split(samples, train_size=train_size, shuffle=True, stratify=[lb for _, _, lb in samples])
		else:
			train_samples, eval_samples = samples, []
			random.shuffle(train_samples)
		if swap_aug:
			self.samples_aug_swap_order(train_samples)
			random.shuffle(train_samples)
		os.makedirs(save_folder, exist_ok=True)
		self.save_samples_as_csv(train_samples, os.path.join(save_folder, 'train.csv'))
		SampleExplainer(train_samples).explain_save(os.path.join(save_folder, 'train_info.json'))
		if has_eval:
			self.save_samples_as_csv(eval_samples, os.path.join(save_folder, 'eval.csv'))
			SampleExplainer(eval_samples).explain_save(os.path.join(save_folder, 'eval_info.json'))


	@timer
	def save_samples_as_csv(self, samples, csv_path):
		"""
		Args:
			samples (list): [(sent1, sent2, label), ...]
			csv_path (str)
		"""
		print('saving: ', csv_path)
		pd.DataFrame(
			[{'text_a': text_a, 'text_b': text_b, 'label': label} for text_a, text_b, label in samples],
			columns=['text_a', 'text_b', 'label']).to_csv(csv_path, index=False)
		if len(samples) > 100000:
			csv_path = os.path.splitext(csv_path)[0] + '-small.csv'
			samples = random.sample(samples, 10000)
			pd.DataFrame(
				[{'text_a':text_a, 'text_b':text_b, 'label':label} for text_a, text_b, label in samples],
				columns=['text_a', 'text_b', 'label']).to_csv(csv_path, index=False)


	@timer
	def samples_aug_swap_order(self, samples):
		aug_samples = [(text_b, text_a, label) for text_a, text_b, label in samples]
		samples.extend(aug_samples)


	@timer
	def select_pos_pairs(self, syn_term_lists, max_seq_len=None, include_self=False, from_hpo_only=False):
		"""
		Args:
			syn_term_sets (list): [syn_terms1, syn_terms2, ...]; syn_terms = [str1, str2, ...]
		Returns:
			list: [(str1, str2), ...]
		"""
		def find_std_term(term_list, std_term_set):
			for i, term in enumerate(term_list):
				if term in std_term_set:
					return i, term
			ret_idx = random.choice(list(range(len(term_list))))
			print('Random choice {} as std term'.format(term_list[ret_idx]))
			return ret_idx, term_list[ret_idx]

		pos_pairs = []
		if from_hpo_only:
			std_term_set = set(self.get_hpo_reader().get_cns_list())
			for syn_terms in syn_term_lists:
				std_idx, std_term = find_std_term(syn_terms, std_term_set)
				for i in range(len(syn_terms)):
					if i == std_idx:
						continue
					pos_pairs.append((std_term, syn_terms[i]))
		else:
			for syn_terms in syn_term_lists:
				term_num = len(syn_terms)
				for i in range(term_num):
					for j in range(i if include_self else (i+1), term_num):
						pos_pairs.append((syn_terms[i], syn_terms[j]))
		pos_pairs = self.max_seq_length_filter(pos_pairs, max_seq_len)
		return unique_pairs(pos_pairs)


	@timer
	def select_neg_pairs(self, syn_term_lists, neg_num, min_jaccard=0.0, cpu_use=12, chunk_size=200,
			max_seq_len=None, from_hpo_only=False):
		"""
		Args:
			syn_term_lists (list): [syn_terms1, syn_terms2, ...]; syn_terms = [str1, str2, ...]
			neg_num (int)
			min_jaccard (float)
			cpu_use (int)
			chunk_size: (int)
		Returns:
			list: [(str1, str2), ...]
		"""
		if neg_num == 0:
			return []
		neg_pairs = []
		if from_hpo_only:
			term_to_concept_id = {t: i for i, syn_terms in enumerate(syn_term_lists) for t in syn_terms}
			cns_list = self.get_hpo_reader().get_cns_list()
			while len(neg_pairs) < neg_num:
				terms1 = np.random.choice([syn_term for syn_terms in syn_term_lists for syn_term in syn_terms], neg_num + 1000)
				terms2 = np.random.choice(cns_list, neg_num + 1000)
				neg_pairs.extend([(t1, t2) for t1, t2 in zip(terms1, terms2) if term_to_concept_id.get(t1, -1) != term_to_concept_id.get(t2, -2)])
				print('negative pairs: {} ({:.4}%)'.format(len(neg_pairs), len(neg_pairs) * 100. / neg_num))
		else:
			while len(neg_pairs) < neg_num:
				shuffle_ranks = list(range(len(syn_term_lists)))
				random.shuffle(shuffle_ranks)
				pairs = [(str1, str2) for i1, i2 in enumerate(shuffle_ranks) for str1, str2 in itertools.product(syn_term_lists[i1], syn_term_lists[i2])]
				if min_jaccard > 0.0:  # Jaccard filter
					jaccard_sims = cal_jaccard_sim_list(pairs, cpu_use=cpu_use, chunk_size=chunk_size)
					pairs = [p for p, s in zip(pairs, jaccard_sims) if s > min_jaccard]
				pairs = self.max_seq_length_filter(pairs, max_seq_len)
				neg_pairs.extend(pairs)
				neg_pairs = unique_pairs(neg_pairs)
				print('negative pairs: {} ({:.4}%)'.format(len(neg_pairs), len(neg_pairs)*100./neg_num))
		return random.sample(neg_pairs, neg_num)


	@timer
	def preprocess_syn_term_lists(self, syn_term_lists):
		ret = []
		for syn_terms in syn_term_lists:
			ret.append(list(set([t.strip() for t in syn_terms])))
		return ret


	def pairs_to_samples(self, pos_pairs, neg_pairs):
		samples = []
		for text_a, text_b in pos_pairs:
			samples.append((text_a, text_b, self.POS_LABEL))
		for text_a, text_b in neg_pairs:
			samples.append((text_a, text_b, self.NEG_LABEL))
		return samples


	def syn_term_lists_to_samples(self, syn_term_lists, neg_x=1.0, min_jaccard=0.0, cpu_use=12, chunk_size=200,
			max_seq_len=None, include_self=False, neg_from_hpo=False, pos_from_hpo=False):
		pos_pairs = self.select_pos_pairs(syn_term_lists, max_seq_len=max_seq_len,
			include_self=include_self, from_hpo_only=pos_from_hpo)
		neg_pairs = self.select_neg_pairs(syn_term_lists, int(len(pos_pairs) * neg_x), min_jaccard,
			cpu_use, chunk_size, max_seq_len=max_seq_len, from_hpo_only=neg_from_hpo)
		return self.pairs_to_samples(pos_pairs, neg_pairs)


	@timer
	def gen_train_dataset(self, subsets, save_folder=None, train_size=1.0, neg_x=1.0, min_jaccard=0.0,
			cpu_use=12, chunk_size=200, swap_aug=False, max_seq_len=None, include_self=False,
			neg_from_hpo=False, pos_from_hpo=False, del_repeat=False, sample_nums='all'):
		"""
		Args:
			subsets (list): [name1, name2, ...]; name = 'hpo_source_syn' | 'cui_source_syn' | 'cui_source_bg_syn' | 'chip'
			neg_x (float or list): The number of negative pairs will be neg_x times as many as positive pairs
			neg_from_hpo (bool or list)
			pos_from_hpo (bool or list)
			sample_nums (str or list)
		"""
		def process_code_to_syn_terms(code_to_syn_terms):
			for code, syns in list(code_to_syn_terms.items()):
				if len(syns) == 0:
					del code_to_syn_terms[code]

		def process_arg(arg):
			if isinstance(arg, list):
				assert len(arg) == len(subsets)
				return arg
			return [arg] * len(subsets)
		def print_samples_info(mark, samples):
			info = SampleExplainer(samples).explain()
			info = {k: v for k, v in info.items() if k in ['SAMPLE_NUM', 'LABEL_COUNT']}
			print('{}: {}'.format(mark, info))
		def sample_partial(samples, sample_num):
			if sample_num == 'all':
				return samples
			ranks = np.random.choice(list(range(len(samples))), size=sample_num)
			return [samples[idx] for idx in ranks]

		save_folder = save_folder or os.path.join(
			DATA_PATH, 'preprocess', 'dataset', '{}-len{}-self{}-neg{}-sim{}-swap{}-neg_hpo{}-pos_hpo{}-train{}-spn{}-delr{}'.format(
				'-'.join(sorted(subsets)), max_seq_len, include_self, neg_x, min_jaccard, swap_aug, neg_from_hpo,
				pos_from_hpo, train_size, sample_nums, del_repeat))
		neg_from_hpo = process_arg(neg_from_hpo)
		pos_from_hpo = process_arg(pos_from_hpo)
		neg_x = process_arg(neg_x)
		sample_nums = process_arg(sample_nums)
		samples = []
		for name, nx, nfh, pfh, spn in zip(subsets, neg_x, neg_from_hpo, pos_from_hpo, sample_nums):
			if name == 'hpo_exact':
				hpo_terms = self.get_hpo_reader().get_cns_list()
				sub_samples = [(hpo_term, hpo_term, self.POS_LABEL) for hpo_term in hpo_terms]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_source_syn':
				hpo_to_syn_terms = self.get_syn_dict_reader().get_hpo_to_source_syn_terms()
				process_code_to_syn_terms(hpo_to_syn_terms)
				sub_samples = self.syn_term_lists_to_samples(
					list(hpo_to_syn_terms.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=pfh)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_source_dict_syn':
				hpo_to_syn_terms = self.get_syn_dict_reader().get_hpo_to_source_dict_syn_terms()
				process_code_to_syn_terms(hpo_to_syn_terms)
				sub_samples = self.syn_term_lists_to_samples(
					list(hpo_to_syn_terms.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=pfh)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_source_bg_syn':
				hpo_to_syn_terms = self.get_syn_dict_reader().get_hpo_to_syn_terms_with_bg_evaluate()
				# process_hpo_to_syn_terms(hpo_to_syn_terms)
				sub_samples = self.syn_term_lists_to_samples(
					list(hpo_to_syn_terms.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=pfh)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_to_source_bg_dict_syn':
				hpo_to_syn_terms = self.get_syn_dict_reader().get_hpo_to_source_bg_dict_syn_terms()
				sub_samples = self.syn_term_lists_to_samples(
					list(hpo_to_syn_terms.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=pfh)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_to_neg_sim':
				hpo_to_syns, mark = self.get_syn_dict_reader().get_hpo_to_source_syn_terms(), 'source'
				sim_from, sim_to = 0.2, 0.5
				neg_samples = self.get_syn_dict_reader().get_hpo_neg_with_jaccard(
					min_sim=sim_from, hpo_to_syns=hpo_to_syns, mark=mark, cpu_use=20, select_from_sim=0.2)
				sub_samples = [(ns[0], ns[1], self.NEG_LABEL) for ns in neg_samples if ns[2] >= sim_from and ns[2] <= sim_to]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_parent_child':
				parent_child_pairs = self.get_hpo_reader().get_parent_child_pairs()
				hpo_to_cns = self.get_hpo_reader().get_hpo_to_cns()
				sub_samples = [(hpo_to_cns[hpo1], hpo_to_cns[hpo2], self.POS_LABEL) for hpo1, hpo2 in parent_child_pairs if hpo1 in hpo_to_cns and hpo2 in hpo_to_cns]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_brothers':
				neg_pairs = self.get_hpo_reader().get_brother_pairs()
				sub_samples = [(text_a, text_b, self.NEG_LABEL) for text_a, text_b in neg_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_source_syn':
				sub_samples = self.syn_term_lists_to_samples(
					list(self.get_syn_dict_reader().get_cui_to_source_syn_terms().values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_source_dict_syn':
				sub_samples = self.syn_term_lists_to_samples(
					list(self.get_syn_dict_reader().get_cui_to_source_dict_syn_terms().values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_source_bg_syn':
				cui_to_syns = self.get_syn_dict_reader().get_cui_to_syn_terms_with_bg_evaluate()
				sub_samples = self.syn_term_lists_to_samples(
					list(cui_to_syns.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_source_bg_dict_syn':
				cui_to_syns = self.get_syn_dict_reader().get_cui_to_source_bg_dict_syn_terms()
				sub_samples = self.syn_term_lists_to_samples(
					list(cui_to_syns.values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_dict_syn':
				sub_samples = self.syn_term_lists_to_samples(
					list(self.get_syn_dict_reader().get_cui_to_dict_syn_terms().values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'mxh900k':
				sub_samples = []
				for text_a, text_b, label in pd.read_csv(os.path.join(DATA_PATH, 'preprocess', 'dataset', 'mxh', 'train.csv')).values:
					sub_samples.append((text_a, text_b, str(label)))
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'mxh_1.8m_result4':
				sub_samples = []
				for text_a, text_b, label in pd.read_csv(os.path.join(DATA_PATH, 'preprocess', 'dataset', 'mxh_result4', 'train.csv')).values:
					sub_samples.append((text_a, text_b, str(label)))
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hit':
				sub_samples = self.syn_term_lists_to_samples(
					list(self.get_hit_reader().get_code_to_syns().values()),
					nx, min_jaccard, cpu_use, chunk_size, max_seq_len=max_seq_len,
					include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'antonym':
				neg_pairs = self.get_antonym_reader().get_term_pairs()
				sub_samples = [(text_a, text_b, self.NEG_LABEL) for text_a, text_b in neg_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_antonym':
				antonyms_pairs = self.get_hpo_reader().get_antonyms_pairs()
				pos_pairs, neg_pairs = [], []
				for terms1, terms2 in antonyms_pairs:
					pos_pairs.extend(list(itertools.product(terms1, terms1)))
					pos_pairs.extend(list(itertools.product(terms2, terms2)))
					neg_pairs.extend(list(itertools.product(terms1, terms2)))
				sub_samples = self.pairs_to_samples(pos_pairs, neg_pairs)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_ant_enhance':
				neg_pairs = self.get_hpo_reader().get_ant_enhance_pairs()
				sub_samples = [(text_a, text_b, self.NEG_LABEL) for text_a, text_b in neg_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_neg_detect_enhance':
				neg_pairs = self.get_hpo_reader().get_neg_detect_enhance_pairs()
				sub_samples = [(text_a, text_b, self.NEG_LABEL) for text_a, text_b in neg_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'chip':
				sub_samples = self.syn_term_lists_to_samples(
					ChipReader().get_all_pairs(), nx, min_jaccard, cpu_use, chunk_size,
					max_seq_len=max_seq_len, include_self=include_self, neg_from_hpo=nfh, pos_from_hpo=False)
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'symptom_context_pos':
				from bert_syn.core.span_generator import PUMC2000CorpusGenerator
				code_to_syns = self.get_syn_dict_reader().get_hpo_to_source_syn_terms()
				words = [t for syns in code_to_syns.values() for t in syns]
				corpus = PUMC2000CorpusGenerator().get_corpus() # sent_split_pattern=re.compile(r'[，。！？；\n\r\t]')
				pos_pairs = gen_context_pos_pairs(words, corpus, matcher='exact')
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'random_ctx_pos_pairs':
				from bert_syn.core.span_generator import PUMC2000CorpusGenerator
				code_to_syns = self.get_syn_dict_reader().get_hpo_to_source_syn_terms()
				words = [t for syns in code_to_syns.values() for t in syns]
				corpus = PUMC2000CorpusGenerator().get_corpus()  # sent_split_pattern=re.compile(r'[，。！？；\n\r\t]')
				pos_pairs = gen_random_ctx_pos_pairs(words, corpus, spn)
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'symptom_combine_pos_pairs':
				pos_pairs = get_symptom_combine_pos_pairs(spn)
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'anatomy_neg':
				neg_pairs = get_anatomy_neg_pairs()
				sub_samples = [(text_a, text_b, self.NEG_LABEL) for text_a, text_b in neg_pairs]
				sub_samples = sample_partial(sub_samples, spn)
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'cui_symptom_disease_pos':
				cui_to_syns = self.get_syn_dict_reader().get_cui_to_source_syn_terms()
				pos_pairs = get_cui_select_pos_pairs(cui_to_syns, spn)
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_def_pos':
				pos_pairs = get_hpo_def_pos_pairs()
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			elif name == 'hpo_eng_pos':
				pos_pairs = get_hpo_eng_pos_pairs()
				sub_samples = [(text_a, text_b, self.POS_LABEL) for text_a, text_b in pos_pairs]
				print_samples_info(name, sub_samples)
				samples.extend(sub_samples)
			else:
				raise RuntimeError('Unknown subset name: {}'.format(name))
		if del_repeat:
			samples = unique_lists(samples)
		print_samples_info('Final smaples:', samples)
		os.makedirs(save_folder, exist_ok=True)
		self.split_and_save_samples(samples, save_folder, train_size, swap_aug,)


	@timer
	def gen_pumc_dataset(self, data_type='test', save_folder=None, sample_num=None):
		save_folder = save_folder or os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', data_type)
		os.makedirs(save_folder, exist_ok=True)
		ehr_hpo_pairs = self.get_pumc_reader().get_doc_tag_pairs(data_type)    # [(ehr_term, hpo_term), ...]
		ehr_to_hpo_texts = OrderedDict()
		for ehr_term, hpo_term in ehr_hpo_pairs:
			ehr_term = ehr_term.strip()
			if len(ehr_term) < 2:
				continue
			if not contain_cns(ehr_term):
				continue
			if re.search('(mol/|ml/|/h|/ul|/mL|/ml|/L|↑|↓|[a-zA-Z0-9]\s*（?\(?\+)', ehr_term) is not None:
				continue
			dict_set_add(ehr_term, hpo_term, ehr_to_hpo_texts)
		for ehr_term in ehr_to_hpo_texts:
			ehr_to_hpo_texts[ehr_term] = list(ehr_to_hpo_texts[ehr_term])
		if sample_num is not None:
			sample_ehr_terms = random.sample(ehr_to_hpo_texts.keys(), sample_num)
			ehr_to_hpo_texts = {t: ehr_to_hpo_texts[t] for t in sample_ehr_terms}

		mark = sample_num if sample_num is not None else "all"
		json.dump(
			list(ehr_to_hpo_texts.keys()),
			open(os.path.join(save_folder, f'ehr_terms_{mark}.json'), 'w'),
			indent=2, ensure_ascii=False)
		json.dump(
			ehr_to_hpo_texts,
			open(os.path.join(save_folder, f'ehr_to_true_texts_{mark}.json'), 'w'),
			indent=2, ensure_ascii=False)
		self.gen_mapping_to_all_hpo_dataset(
			[(ehr_term, hpo_term) for ehr_term, hpo_terms in ehr_to_hpo_texts.items() for hpo_term in hpo_terms],
			os.path.join(save_folder, f'{data_type}_{mark}.csv')
		)


	def term_list_json_to_csv(self, json_path, csv_path=None):
		csv_path =csv_path or (os.path.splitext(json_path)[0] + '.csv')
		term_list = json.load(open(json_path))
		self.save_samples_as_csv([(term, '', -1) for term in term_list], csv_path)


	def gen_mapping_to_all_hpo_dataset(self, raw_true_pairs, save_csv):
		cns_list = list(set(self.get_hpo_reader().get_cns_list()))
		raw_to_true_texts = {}
		for raw_term, true_term in raw_true_pairs:
			dict_set_add(raw_term, true_term, raw_to_true_texts)
		samples = []
		raw_term_set = {raw_term for raw_term, _ in raw_true_pairs}
		for raw_term in raw_term_set:
			true_text_set = raw_to_true_texts[raw_term]
			samples.extend([(raw_term, cns_term, self.POS_LABEL if cns_term in true_text_set else self.NEG_LABEL) for cns_term in cns_list])
		self.save_samples_as_csv(samples, save_csv)
		info_json = os.path.splitext(save_csv)[0] + '-info.json'
		SampleExplainer(samples).explain_save(info_json)


	def random_select_predict_result(self, src_csv, sample_num=None, span_texts=None):
		save_folder = os.path.splitext(src_csv)[0] + '-random-sample'
		os.makedirs(save_folder, exist_ok=True)
		df = pd.read_csv(src_csv)
		if span_texts is None:
			assert sample_num is not None
			sample_ranks = random.sample(list(range(len(df))), sample_num)
			span_texts = list(df.iloc[sample_ranks]['text_a'])
		for span_text in span_texts:
			print('Select:', span_text)
			sub_df = df[df['text_a'] == span_text]
			sub_df.sort_values('label', ascending=False).to_csv(os.path.join(save_folder, f'{span_text.replace("/", "_")}.csv'))


if __name__ == '__main__':
	def test_dict():
		"""generate dict
		"""

		syn_reader = SynDictReader()
		hpo_to_syns, mark = syn_reader.get_hpo_to_source_bg_dict_syn_terms(), 'bg_dict'
		json.dump(hpo_to_syns, open(os.path.join(syn_reader.SAVE_FOLDER, 'hpo_to_source_bg_dict_syn_terms.json'), 'w'), indent=2, ensure_ascii=False)


	def run_dh_others():
		"""
		"""
		pass



	def gen_train_data():
		"""generate train datasets
		"""
		pass
		dh = DataHelper()
		max_seq_len = 64

		dh.gen_train_dataset(
			subsets = ['antonym', 'hpo_eng_pos', 'hpo_ant_enhance', 'hpo_source_dict_syn', 'hpo_to_source_bg_dict_syn', 'hpo_parent_child', 'chip'],  # 'hpo_to_neg_sim'
			neg_x=[None, None, None, 20.0, 20.0, None, 0.0],
			# sample_nums=[10000]+['all']*7,
			max_seq_len=max_seq_len, include_self=False, swap_aug=False, neg_from_hpo=True, pos_from_hpo=True,
			save_folder=os.path.join(DATA_PATH, 'preprocess', 'dataset', 'AN_Hpo_N_SD20_BGD20_PC_C0-3-New_CHPO')	# 命名与subset对应
		)

	gen_train_data()



