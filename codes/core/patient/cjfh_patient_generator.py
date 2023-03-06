

import os
import re
import json
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd

from core.utils.constant import DATA_PATH, RESULT_PATH, JSON_FILE_FORMAT
from core.reader import HPOReader, HPOFilterDatasetReader
from core.reader.umls_reader import UMLSReader
from core.utils.utils import get_file_list, dict_list_extend, dict_list_add, unique_list, split_path, delete_redundacy, check_load
from core.explainer.dataset_explainer import LabeledDatasetExplainer
from core.text_handler.text_searcher import TextSearcher
from core.patient.patient_generator import PatientGenerator

# MSA: MULTIPLE SYSTEM ATROPHY; multiple system atrophy;
#   MSA-A: SDS; Shy–Drager syndrome;
#   MSA-C:  OPCA; Sporadic olivopontocerebellar atrophy; "C" = cerebellar dysfunction subtype
#   MSA-P: ; SND; Striatonigral degeneration; "P" = parkinsonian subtype
# SCA: SPINOCEREBELLAR ATAXIA; spinocerebellar ataxia;


DAIG_MAP_DICT = {
	'MSA': {
		'OMIM:146500': 'MSA1', # MULTIPLE SYSTEM ATROPHY 1, SUSCEPTIBILITY TO; MSA1
		'ORPHA:102': 'MSA', # Multiple system atrophy (Disorder); OMIM:146500
		'ORPHA:227510': 'MSA-C', # Multiple system atrophy, cerebellar type (Subtype of disorder)
		'ORPHA:98933': 'MSA-P', # Multiple system atrophy, parkinsonian type (Subtype of disorder)
		'CCRD:77': 'MSA', # Multiple system atrophy; MSA
	},
	'SCA': {   # searching keyword: SPINOCEREBELLAR ATAXIA
		'OMIM:164400': 'SCA1',
		'OMIM:183090': 'SCA2',
		'OMIM:109150': 'SCA3',
		'OMIM:600223': 'SCA4',
		'OMIM:600224': 'SCA5',
		'OMIM:183086': 'SCA6',
		'OMIM:164500': 'SCA7',
		'OMIM:608768': 'SCA8',
		'OMIM:612876': 'SCA9',
		'OMIM:603516': 'SCA10',
		'OMIM:604432': 'SCA11',
		'OMIM:604326': 'SCA12',
		'OMIM:605259': 'SCA13',
		'OMIM:605361': 'SCA14',
		'OMIM:606658': ['SCA15', 'SCA16'],
		'OMIM:607136': 'SCA17',
		'OMIM:607458': 'SCA18',
		'OMIM:607346': ['SCA19', 'SCA22'],
		'OMIM:608687': 'SCA20',
		'OMIM:607454': 'SCA21',
		'OMIM:610245': 'SCA23',
		'OMIM:608703': 'SCA25',
		'OMIM:609306': 'SCA26',
		'OMIM:609307': 'SCA27',
		'OMIM:610246': 'SCA28',
		'OMIM:117360': 'SCA29',
		'OMIM:613371': 'SCA30',
		'OMIM:117210': 'SCA31',
		'OMIM:613909': 'SCA32',
		'OMIM:133190': 'SCA34',
		'OMIM:613908': 'SCA35',
		'OMIM:614153': 'SCA36',
		'OMIM:615945': 'SCA37',
		'OMIM:615957': 'SCA38',
		'OMIM:616053': 'SCA40',
		'OMIM:616410': 'SCA41',
		'OMIM:616795': 'SCA42',
		'OMIM:617018': 'SCA43',

		'OMIM:606002': ['SCAR1', 'SCAN2'],
		'OMIM:213200': 'SCAR2',
		'OMIM:271250': 'SCAR3',
		'OMIM:607317': ['SCAR4', 'SCA24'],
		'OMIM:608029': 'SCAR6',
		'OMIM:609270': 'SCAR7',
		'OMIM:610743': 'SCAR8',
		'OMIM:612016': 'SCAR9',
		'OMIM:613728': 'SCAR10',
		'OMIM:614229': 'SCAR11',
		'OMIM:614322': 'SCAR12',
		'OMIM:614831': 'SCAR13',
		'OMIM:615386': 'SCAR14',
		'OMIM:615705': 'SCAR15',
		'OMIM:615768': 'SCAR16',
		'OMIM:616127': 'SCAR17',
		'OMIM:616204': 'SCAR18',
		'OMIM:616291': 'SCAR19',
		'OMIM:616354': 'SCAR20',
		'OMIM:616719': 'SCAR21',
		'OMIM:616948': 'SCAR22',
		'OMIM:616949': 'SCAR23',
		'OMIM:617133': 'SCAR24',

		'OMIM:607250': 'SCAN1',
		'OMIM:618387': 'SCAN3',

		'OMIM:302500': 'SCAX1',
		'OMIM:302600': 'SCAX2',
		'OMIM:301790': 'SCAX3',
		'OMIM:301840': 'SCAX4',
		'OMIM:300703': 'SCAX5',

		'OMIM:301310': 'ASAT', # ANEMIA, SIDEROBLASTIC, AND SPINOCEREBELLAR ATAXIA;

		'ORPHA:64753': ['SCAR1', 'SCAN2'],
		'ORPHA:1170': 'SCAR2',
		'ORPHA:95433': 'SCAR3',
		'ORPHA:95434': 'SCAR4',
		'ORPHA:83472': 'SCAR5',
		'ORPHA:284332': 'SCAR6',
		'ORPHA:284324': 'SCAR7',
		'ORPHA:88644': 'SCAR8',
		'ORPHA:139485': 'SCAR9',
		'ORPHA:284289': 'SCAR10',
		'ORPHA:284271': 'SCAR11',
		'ORPHA:284282': 'SCAR12',
		'ORPHA:324262': 'SCAR13',
		'ORPHA:352403': 'SCAR14',
		'ORPHA:404499': 'SCAR15',
		'ORPHA:412057': 'SCAR16',
		'ORPHA:453521': 'SCAR17',
		'ORPHA:363432': 'SCAR18',
		'ORPHA:448251': 'SCAR19',
		'ORPHA:397709': 'SCAR20',
		'ORPHA:466794': 'SCAR21',
		'ORPHA:404493': 'SCAR23',

		'ORPHA:98755': 'SCA1',
		'ORPHA:98756': 'SCA2',
		'ORPHA:98757': 'SCA3',
		'ORPHA:276238': 'SCA3', # Joseph type; Subtype of disorder
		'ORPHA:276241': 'SCA3', # Thomas type; Subtype of disorder
		'ORPHA:276244': 'SCA3', # Machado type; Subtype of disorder
		'ORPHA:98765': 'SCA4',
		'ORPHA:98766': 'SCA5',
		'ORPHA:98758': 'SCA6',
		'ORPHA:94147': 'SCA7',
		'ORPHA:98760': 'SCA8',
		'ORPHA:98761': 'SCA10',
		'ORPHA:98767': 'SCA11',
		'ORPHA:98762': 'SCA12',
		'ORPHA:98768': 'SCA13',
		'ORPHA:98763': 'SCA14',
		'ORPHA:98769': ['SCA15', 'SCA16'],
		'ORPHA:98770': 'SCA16',
		'ORPHA:98759': 'SCA17',
		'ORPHA:98771': 'SCA18',
		'ORPHA:98772': ['SCA19', 'SCA22'],
		'ORPHA:101110': 'SCA20',
		'ORPHA:98773': 'SCA21',
		'ORPHA:101107': 'SCA22',
		'ORPHA:101108': 'SCA23',
		'ORPHA:101111': 'SCA25',
		'ORPHA:101112': 'SCA26',
		'ORPHA:98764': 'SCA27',
		'ORPHA:101109': 'SCA28',
		'ORPHA:208513': 'SCA29',
		'ORPHA:211017': 'SCA30',
		'ORPHA:217012': 'SCA31',
		'ORPHA:276183': 'SCA32',
		'ORPHA:1955': 'SCA34',
		'ORPHA:276193': 'SCA35',
		'ORPHA:276198': 'SCA36',
		'ORPHA:363710': 'SCA37',
		'ORPHA:423296': 'SCA38',
		'ORPHA:423275': 'SCA40',
		'ORPHA:458798': 'SCA41',
		'ORPHA:458803': 'SCA42',
		'ORPHA:497764': 'SCA43',

		'ORPHA:2802': 'X-linked sideroblastic anemia and spinocerebellar ataxia', # OMIM:301310
		'ORPHA:1175': 'SCAX1', # X-linked progressive cerebellar ataxia; OMIM:302500; SCAX1
		'ORPHA:85297': 'SCAX3',
		'ORPHA:85292': 'SCAX4',
		'ORPHA:314978': 'SCAX5', # X-linked non progressive cerebellar ataxia; OMIM:300703; SCAX5

		'ORPHA:94124': 'SCAN1',
		'ORPHA:254881': 'SCAE',
		'ORPHA:1185': 'Spinocerebellar ataxia-dysmorphism syndrome',
		'ORPHA:2074': 'Spinocerebellar ataxia-amyotrophy-deafness syndrome',
		'ORPHA:1186': 'IOSCA', # Infantile onset spinocerebellar ataxia

		'CCRD:111': 'SCA', # Spinocerebellar ataxia; SCA
	}
}


class CJFHPatientGenerator(PatientGenerator):
	# SNOMED Patients
	def __init__(self, hpo_reader=HPOReader()):
		super(CJFHPatientGenerator, self).__init__(hpo_reader)
		self.INPUT_PATIENT_JSON = os.path.join(DATA_PATH, 'raw', 'CJFH', 'patients_v1.5.1.json')
		self.INPUT_PATIENT_SNOMED_FOLDER = os.path.join(DATA_PATH, 'raw', 'CJFH', 'msa-ann')
		self.DIAG_STR_TO_DIS_CODES_CSV = os.path.join(DATA_PATH, 'raw', 'CJFH', 'diag_str_to_dis_codes.csv')

		self.OUTPUT_PATIENT_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'CJFH')
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)
		self.OUTPUT_TEXT_PATIENT_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'text_patients.json')
		self.OUTPUT_TEXT_PID_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'text_pids.json')
		self.text_patients, self.text_pids = None, None
		self.SNOMED2HPO_PATIENT_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'snomed2hpo_patients.json')
		self.SNOMED2HPO_PID_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'snomed2hpo_pids.json')
		self.SNOMED2HPO_PATIENT_STATITICS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'snomed2hpo_patients_statistics.json')
		self.snomed2hpo_patients, self.snomed2hpo_pids = None, None

		self.umls_reader = UMLSReader()
		self.NOUSE_BRACKET_PATTERN = None


	def get_no_use_bracket_pattern(self):
		if self.NOUSE_BRACKET_PATTERN is None:
			lb = ['\(', '（']
			rb = ['\)', '）']
			bracket_terms = [
				'日期、相应变化及其诱因', '工作性质', '家系图见背面', '近亲婚育史', '有无畸形',
				'理解判断力、定向力、记忆力、计算力、书写、言语情况和情感', '震颤，手足徐动，舞蹈动作等',
				'排汗、皮脂溢出、性功能、便秘、四肢循环障碍',
			]
			pattern = '[%s]{1}(%s){1}[%s]{1}' % (''.join(lb), '|'.join(bracket_terms), ''.join(rb))
			self.NOUSE_BRACKET_PATTERN =  re.compile(pattern)
		return self.NOUSE_BRACKET_PATTERN


	def get_pattern_map_dis_info(self):
		"""
		Returns:
			dict: {pattern: {'DIAGNOSIS': dis_code, 'LEVEL': int}}
		"""
		return {
			'多系统萎缩|MSA|Shy-Drager': {'DIAGNOSIS': list(DAIG_MAP_DICT['MSA'].keys()), 'LEVEL':0},  # MSA
			'MSA-.*?C':{'DIAGNOSIS': ['ORPHA:227510'], 'LEVEL':1},  # MSA-C
			'MSA-.*?P':{'DIAGNOSIS': ['ORPHA:98933'], 'LEVEL':1},  # MSA-P
			'SCA|脊髓小脑共济失调|遗传性共济失调': {'DIAGNOSIS': list(DAIG_MAP_DICT['SCA'].keys()), 'LEVEL':0},  # SCA
			'SCA1':{'DIAGNOSIS': ['OMIM:164400'], 'LEVEL':1},
			'SCA2':{'DIAGNOSIS': ['OMIM:183090'], 'LEVEL':1},
			'SCA3':{'DIAGNOSIS': ['OMIM:109150'], 'LEVEL':1},
			'SCA6':{'DIAGNOSIS': ['OMIM:183086'], 'LEVEL':1},
			'SCA7':{'DIAGNOSIS': ['OMIM:164500'], 'LEVEL':1},
			'SCA17':{'DIAGNOSIS': ['OMIM:607136'], 'LEVEL':1},
		}


	def get_remove_pattern(self):
		return '患者之女|患者之子|帕金森综合症|帕金森综合征|后代|除外|待诊|待查|可疑|可能|((\?|？).*(\?|？))'


	def get_diag_str(self, pa_dict):
		diag = ''
		if '初步诊断' in pa_dict:
			diag = pa_dict['初步诊断'].strip()
		if '拟诊' in pa_dict:
			diag = pa_dict['拟诊'.strip()]
		return diag


	@check_load('snomed2hpo_patients', 'SNOMED2HPO_PATIENT_JSON', JSON_FILE_FORMAT)
	def get_snomed2hpo_patients(self):
		patients, keep_pids = self._gen_snomed2hpo_patients()
		return patients

	@check_load('snomed2hpo_pids', 'SNOMED2HPO_PID_JSON', JSON_FILE_FORMAT)
	def get_snomed2hpo_pids(self):
		patients, pids = self._gen_snomed2hpo_patients()
		return pids


	def _gen_snomed2hpo_patients(self):
		pids = self.get_all_pids()
		pid2hpolist = self.get_pid_to_snomed2hpo_list()
		pid2discodes = self.get_pid_to_dis_codes()
		patients, keep_pids = [], []
		for pid in pids:
			hpo_list, dis_codes = pid2hpolist[pid], pid2discodes[pid]
			patients.append((hpo_list, dis_codes))
			keep_pids.append(pid)
		json.dump(keep_pids, open(self.SNOMED2HPO_PID_JSON, 'w'), indent=2)
		json.dump(patients, open(self.SNOMED2HPO_PATIENT_JSON, 'w'), indent=2)
		return patients, pids


	@check_load('text_patients', 'OUTPUT_TEXT_PATIENT_JSON', JSON_FILE_FORMAT)
	def get_text_patients(self):
		patients, keep_pids = self._gen_text_patients()
		return patients


	@check_load('text_pids', 'OUTPUT_TEXT_PID_JSON', JSON_FILE_FORMAT)
	def get_text_pids(self):
		patients, pids = self._gen_text_patients()
		return pids


	def _gen_text_patients(self):
		pids = self.get_all_pids()
		pid2text = self.get_pid_to_patient_text()
		pid2discodes = self.get_pid_to_dis_codes()
		patients, keep_pids = [], []
		for pid in pids:
			text, dis_codes = pid2text[pid], pid2discodes[pid]
			patients.append((text, dis_codes))
			keep_pids.append(pid)
		json.dump(keep_pids, open(self.OUTPUT_TEXT_PID_JSON, 'w'), indent=2)
		json.dump(patients, open(self.OUTPUT_TEXT_PATIENT_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients, keep_pids


	def get_all_pids(self):
		pa_info_list = json.load(open(self.INPUT_PATIENT_JSON))
		return sorted([pa_info['id'] for pa_info in pa_info_list])


	def get_pid_to_pa_info(self):
		"""
		Returns:
			dict: {pid: pa_info}
		"""
		pa_info_list = json.load(open(self.INPUT_PATIENT_JSON))
		return {pa_info['id']: pa_info for pa_info in pa_info_list}


	def get_pid_to_snomed2hpo_list(self):
		"""
		Returns:
			list: {pid: [hpo_code1, ...]}
		"""
		cui_to_hpo_list = self.umls_reader.get_cui_to_hpo_list()
		snomed_to_hpo_list = self.get_snomed_to_hpo_list()
		ann_path_list = self.get_ann_path_list()
		ret_dict = {}
		for ann_path in ann_path_list:
			pid = int(split_path(ann_path)[1])
			hpo_codes, _ = self.get_hpo_and_pos_from_ann(ann_path, snomed_to_hpo_list, cui_to_hpo_list)
			ret_dict[pid] = unique_list(hpo_codes)
		return ret_dict


	def diag_str_to_dis_codes(self, diag_str, remove_pattern, pattern_map_dis_info):
		match_dis_codes, match_patterns = [], []
		if not re.search(remove_pattern, diag_str):
			for pattern, dis_info in pattern_map_dis_info.items():
				if re.search(pattern, diag_str) is not None:
					match_dis_codes.extend(dis_info['DIAGNOSIS'])
					match_patterns.append(pattern)
			print(diag_str, match_patterns)
		return list(set(match_dis_codes))


	def get_pid_to_diag_str(self):
		pa_info_list = json.load(open(self.INPUT_PATIENT_JSON))
		return {pa_info['id']: self.get_diag_str(pa_info) for pa_info in pa_info_list}


	def get_pid_to_dis_codes(self):
		"""
		Returns:
			dict: {pid: [dis_code1, ...]}
		"""
		pa_info_list = json.load(open(self.INPUT_PATIENT_JSON))
		remove_pattern = self.get_remove_pattern()
		pattern_map_dis_info = self.get_pattern_map_dis_info()
		ret_dict = {}
		for pa_info in pa_info_list:
			pid = pa_info['id']
			diag_str = self.get_diag_str(pa_info)
			dis_codes = self.diag_str_to_dis_codes(diag_str, remove_pattern, pattern_map_dis_info)
			ret_dict[pid] = dis_codes
		return ret_dict


	def dis_codes_to_id(self, dis_codes):
		return str(sorted(dis_codes))


	def get_man_search_text_result(self, rank_list=None):
		"""
		Returns:
			list: [searchText, ...]
			list: [search_result, ...]; search_result=([hpo_code, ...], [np.array([begin, end]), ...])
		"""
		cui_to_hpo_list = self.umls_reader.get_cui_to_hpo_list()
		snomed_to_hpo_list = self.get_snomed_to_hpo_list()
		sort_result = TextSearcher().sort_result
		pids = self.get_all_pids()
		ret_search_text, ret_search_result = [], []
		for pid in pids:
			search_hpo_list, search_pos_list = [], []
			txt_path = os.path.join(self.INPUT_PATIENT_SNOMED_FOLDER, '{:04}.txt'.format(pid))
			ann_path = os.path.join(self.INPUT_PATIENT_SNOMED_FOLDER, '{:04}.ann'.format(pid))
			hpo_codes, multi_pos_list = self.get_hpo_and_pos_from_ann(ann_path, snomed_to_hpo_list, cui_to_hpo_list)
			for hpo_code, pos_list in zip(hpo_codes, multi_pos_list):
				search_hpo_list.extend([hpo_code]*len(pos_list))
				search_pos_list.extend([np.array([pos[0], pos[1]]) for pos in pos_list])
			ret_search_result.append(sort_result(search_hpo_list, search_pos_list))
			ret_search_text.append(open(txt_path).read())
		return ret_search_text, ret_search_result


	def get_ann_path_list(self):
		ann_path_list = get_file_list(self.INPUT_PATIENT_SNOMED_FOLDER, lambda path:path.endswith('.ann'))
		_, ann_path_list = zip(*sorted([(int(split_path(path)[1]), path) for path in ann_path_list]))
		return ann_path_list


	def get_txt_path_list(self):
		txt_path_list = get_file_list(self.INPUT_PATIENT_SNOMED_FOLDER, lambda path:path.endswith('.txt'))
		_, txt_path_list = zip(*sorted([(int(split_path(path)[1]), path) for path in txt_path_list]))
		return txt_path_list


	def get_dis_codes_from_str(self, diag_str):
		"""
		Returns:
			list: [dis_code, ...]
		"""
		candi_items, max_level = [], -1  # max_level: use the most specific diagnosis
		for pattern, map_item in self.pattern_map_dis_info.items():
			if re.search(pattern, diag_str) is not None:
				candi_items.append(map_item)
				max_level = map_item['LEVEL'] if map_item['LEVEL'] > max_level else max_level
		dis_codes = []
		for candi_item in candi_items:
			if candi_item['LEVEL'] == 0: #
			# if candi_item['LEVEL'] == max_level: # 530 patients
			# if candi_item['LEVEL'] == max_level and max_level > 0:  # 70 patients
				dis_codes.extend(candi_item['DIAGNOSIS'])
		return unique_list(dis_codes)


	def get_man_ann_patient_text(self):
		"""
		Returns:
			list: [pa_text, ...]
		"""
		path_list = get_file_list(self.INPUT_PATIENT_SNOMED_FOLDER, lambda path: path.endswith('.txt'))
		rank_path_list = sorted([(int(split_path(path)[1]), path) for path in path_list])
		assert list(range(len(path_list))) == [rank for rank, _ in rank_path_list]
		return [open(path).read().strip() for rank, path in rank_path_list]


	def get_pid_to_patient_text(self):
		"""
		Returns:
			dict: {pid: pa_text, ...}
		"""
		ret_dict = {}
		b_pattern = self.get_no_use_bracket_pattern()
		pa_info_list = json.load(open(self.INPUT_PATIENT_JSON))
		for pa_info in pa_info_list:
			pid = pa_info['id']
			pa_text = '\n'.join(self.dict_to_line_list(
				pa_info, remove_key_set={'初步诊断', '拟诊', '家族史', '开出的检查项目', '治疗方案', '医嘱'}, remove_value_set={'-', '（-）'})) # remove_key_set={'家族史', '就诊日期', 'id', '年龄', '收缩压变化', '舒张压变化'}
			pa_text = b_pattern.sub('', pa_text)
			ret_dict[pid] = pa_text
		return ret_dict


	def dict_to_line_list(self, d, remove_key_set=set(), remove_value_set=set()):
		b_pattern = self.get_no_use_bracket_pattern()
		ret_str_list = []
		for k, v in d.items():
			if k in remove_key_set:
				continue
			if isinstance(v, str):
				v = b_pattern.sub('', v).strip()
				if len(v) > 0 and v not in remove_value_set:
					ret_str_list.append('{}:{}'.format(k.strip(), v))
			elif isinstance(v, int) or isinstance(v, float):
				ret_str_list.append('{}:{}'.format(k.strip(), v))
			elif isinstance(v, list):
				v = ';'.join(v).strip()
				if len(v) > 0:
					ret_str_list.append('{}:{}'.format(k, v))
			else:
				assert isinstance(v, dict)
				chd_str_list = self.dict_to_line_list(v, remove_key_set, remove_value_set)
				ret_str_list.extend(['{}.{}'.format(k.strip(), chdStr) for chdStr in chd_str_list])
		return ret_str_list


	def get_hpo_and_pos_from_ann(self, ann_path, snomed_to_hpo_list, cui_to_hpo_list):
		"""
		Returns:
			list: [hpo_code1, ...]
			pos: [[(begin1.1, end1.1), ...], ...]
		"""
		info_list = self.read_ann(ann_path)
		retHpoList, ret_pos_list = [], []
		for item_dict in info_list:
			for code in item_dict['CODE']:
				if self.is_snomed_code(code):
					hpo_list = snomed_to_hpo_list.get(code, [])
				else:
					hpo_list = cui_to_hpo_list.get(code, [])
				matched_hpo_num = len(hpo_list)
				if matched_hpo_num == 0:
					continue
				retHpoList.extend(hpo_list)
				ret_pos_list.extend([item_dict['POSITIONS']]*matched_hpo_num)
		return retHpoList, ret_pos_list


	def read_ann(self, ann_path):
		"""
		Returns:
			list: [{'ENTITY': entity, 'POSITIONS': [(begin, end), ...], 'CODE': [code, ...]}, ];
		"""
		lines = open(ann_path).readlines()
		t_lines = [line for line in lines if line.startswith('T')]
		n_lines = [line for line in lines if line.startswith('N')]
		assert len(t_lines) + len(n_lines) == len(lines)
		TDict = {}  # {'T1': {'ENTITY': entity, 'POSITIONS': [(begin, end), ...], 'CODE': [code, ...]}, ...}
		for line in t_lines:
			T, entityPos, entity_name = line.strip().split('\t')
			_, multi_pos = entityPos.split(' ', maxsplit=1)
			pos_list = []
			for posStr in multi_pos.split(';'):
				begin, end = posStr.split(' ')
				pos_list.append((int(begin), int(end)))
			TDict[T] = {'ENTITY': entity_name.strip(), 'POSITIONS': pos_list}
		for line in n_lines:
			N, refT, owl_term = line.strip().split('\t')
			_, T, code = refT.split(' ')
			assert code.startswith('MSA:')
			dict_list_add('CODE', code[4:].strip(), TDict[T])
		return list(TDict.values())


	def is_snomed_code(self, code):
		return code.find('-') != -1


	def get_snomed_to_hpo_list(self):
		"""
		Returns:
			dict: {snomed_code: [hpo_code1, ...]}
		"""
		mrconso_dict = self.umls_reader.get_mrconso()
		cui_to_hpo_list = self.umls_reader.get_cui_to_hpo_list()
		ret_dict = {}
		for aui, line_dict in mrconso_dict.items():
			if line_dict['SAB'] == 'SNMI':
				snomed_code = line_dict['CODE']
				CUI = line_dict['CUI']
				if CUI in cui_to_hpo_list:
					dict_list_extend(snomed_code, cui_to_hpo_list[CUI], ret_dict)
		for k, v_list in ret_dict.items():
			ret_dict[k] = unique_list(v_list)
		return ret_dict


	def statistic_text_patient(self):
		patients = json.load(open(self.OUTPUT_TEXT_PATIENT_JSON))
		patient_num = len(patients)
		ave_doc_len = sum([len(p[0]) for p in patients]) / patient_num
		ave_diag_num = sum([len(p[1]) for p in patients]) / patient_num
		print('Patient Number = {}\nAverage Doc Length = {}\nAverage Diagnosis Number = {}'.format(patient_num, ave_doc_len, ave_diag_num))


	def statistic_raw(self):
		pa_list = json.load(open(self.INPUT_PATIENT_JSON))
		print('patient number = {}'.format(len(pa_list)))

		# count diagnosis
		counter = Counter()
		for pa_dict in pa_list:
			counter[self.get_diag_str(pa_dict)] += 1
		for diag, count in counter.most_common():
			print(diag, count)


	def gen_diag_str_to_dis_codes(self):
		pa_list = json.load(open(self.INPUT_PATIENT_JSON))
		diag_count_list = Counter([self.get_diag_str(pa_dict) for pa_dict in pa_list]).most_common()
		remove_pattern = self.get_remove_pattern()
		row_infos = []
		for diag, count in diag_count_list:
			match_patterns, match_dis_codes = [], []
			if not re.search(remove_pattern, diag):
				for pattern, dis_info in self.get_pattern_map_dis_info().items():
					if re.search(pattern, diag) is not None:
						match_patterns.append(pattern)
						match_dis_codes.extend(dis_info['DIAGNOSIS'])
			row_infos.append({
				'DIAG_STR': diag,
				'COUNT': count,
				'MATCH_PATTERNS': match_patterns,
				'MATCH_DIS_CODES': match_dis_codes
			})
		pd.DataFrame(row_infos).to_csv(self.DIAG_STR_TO_DIS_CODES_CSV, index=False,
			columns=['DIAG_STR', 'COUNT', 'MATCH_PATTERNS', 'MATCH_DIS_CODES'])


	def get_labels_set_with_all_eq_sources(self, sources):
		"""
		Returns:
			set: {sorted_dis_codes_tuple, ...}; sorted_dis_codes_tuple = (dis_code1, dis_code2, ...)
		"""
		pid_to_dis_codes = self.get_pid_to_dis_codes()
		return set([tuple(sorted(dis_codes)) for dis_codes in pid_to_dis_codes.values() if self.diseases_from_all_sources(dis_codes, sources)])


	def test_pattern(self):
		pa_list = json.load(open(self.INPUT_PATIENT_JSON))
		diag_list = list(set([self.get_diag_str(pa_dict) for pa_dict in pa_list]))
		for diag in diag_list:
			matches = []
			for pattern in self.get_pattern_map_dis_info():
				if re.search(pattern, diag) is not None:
					matches.append(pattern)
			print('{}: {}'.format(diag, matches))


	def test_read_ann(self):
		ann_paths = get_file_list(self.INPUT_PATIENT_SNOMED_FOLDER, lambda path:path.endswith('.ann'))
		for ann_path in ann_paths:
			print(ann_path)
			print(self.read_ann(ann_path))


if __name__ == '__main__':
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	pg = CJFHPatientGenerator(hpo_reader=hpo_reader)
	pg.gen_diag_str_to_dis_codes()

	pg.get_text_patients()


