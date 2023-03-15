

import re
import os
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

from core.utils.utils import check_load, dict_set_update, dict_set_add, dict_list_extend, get_all_descendents, get_file_list
from core.utils.utils import slice_list_with_keep_set, check_load_save, dict_list_add, split_path
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT, TEST_DATA, VALIDATION_DATA
from core.patient.pumc_patient_generator import PUMCPatientGenerator
from core.explainer.dataset_explainer import LabeledDatasetExplainer
from core.explainer import Explainer
from core.reader import HPOReader, HPOFilterDatasetReader, RDReader, OrphanetReader, OMIMReader, CCRDReader
from core.text_handler.term_matcher import ExactTermMatcher
from core.text_handler.max_inv_text_searcher import MaxInvTextSearcher


class PUMC2000PatientGenerator(PUMCPatientGenerator):
	def __init__(self, fields=None, del_diag_hpo=False, mark='', input_folder=None, hpo_reader=HPOReader()):
		rd_reader = RDReader()
		self.rd_dict = rd_reader.get_rd_dict()
		self.source_to_rd = rd_reader.get_source_to_rd()
		super(PUMC2000PatientGenerator, self).__init__(fields, del_diag_hpo, mark=mark, input_folder=input_folder, hpo_reader=hpo_reader)
		matcher = ExactTermMatcher(self.cls_to_terms, syn_dict_name='rd-re', hpo_to_gold_ps=self.cls_to_patterns)
		matcher.std_term_to_hpos = {}
		self.searcher = MaxInvTextSearcher(matcher)
		self.searcher.add_neg_terms(['待除外'])
		self.searcher.del_neg_terms(['除外'])


	def get_rd_to_cns_list(self):
		"""
		Returns:
			dict: {rd_code: [cns_term, ...]}
		"""
		cns_omim_dict = OMIMReader().get_cns_omim()
		cns_orpha_dict = OrphanetReader().get_cns_orpha_dict()
		cns_ccrd_dict = CCRDReader().get_ccrd_dict()
		source_code_to_cns = {
			**{omim_code: info['CNS_NAME'] for omim_code, info in cns_omim_dict.items() if info.get('CNS_NAME', '')},
			**{orpha_code: info['CNS_NAME'] for orpha_code, info in cns_orpha_dict.items() if info.get('CNS_NAME', '')},
			**{ccrd_code: info['CNS_NAME'] for ccrd_code, info in cns_ccrd_dict.items() if info.get('CNS_NAME', '')}
		}
		rd_to_cns_list = {}
		for rd_code, info in self.rd_dict.items():
			for source_code in info['SOURCE_CODES']:
				if source_code in source_code_to_cns:
					dict_set_add(rd_code, source_code_to_cns[source_code], rd_to_cns_list)
		for rd_code in rd_to_cns_list:
			rd_to_cns_list[rd_code] = list(rd_to_cns_list[rd_code])
		return rd_to_cns_list


	def init_pattern_dict(self):
		super(PUMC2000PatientGenerator, self).init_pattern_dict()
		del self.PATTERNS_TO_CLASSES[('线粒体病',)]
		self.PATTERNS_TO_CLASSES_SUPP = {
			('线粒体脑肌病', '眼外肌麻痹\s线粒体病'): ['线粒体脑肌病'],
			('ARVC',): ['致心律失常性右室心肌病'],
			('左心?室(心肌)?致?密化不全',): ['左室致密化不全'],
			('抗\s*LGI1\s*(抗体)?(相关)?脑炎', 'LGI1型脑炎'): ['抗LGI1抗体相关脑炎'],
			('生长激素缺乏症', '矮小症'): ['生长激素缺乏症'],
			('CADASIL',): ['CADASIL'],
			('成人\s*[Ss]till\s*病',): ['成人Still病'],
			('坏疽性脓皮病',): ['坏疽性脓皮病'],
			('白(塞|赛)病', ''): ['白赛病'],
			('Carney',): ['Carney complex'],
			('干燥综合(症|征)',): ['干燥综合征'],
			('Addison.{0,3}病',): ['Addison病'],
			('嗜酸性?粒?细胞性?(胃肠|肠胃)炎', ): ['嗜酸性粒细胞性胃肠炎'],
			('慢性嗜酸性?粒?细胞性?肺炎', ): ['慢性嗜酸性粒细胞性肺炎'],
			('嗜酸性?粒?细胞性?肺炎',): ['嗜酸性粒细胞性肺炎'],
			('ANCA\s*(相关)?性?血管炎',): ['ANCA相关血管炎'],
			('中枢性性早熟',): ['中枢性性早熟'],
			('Good\s*综合征',): ['Good综合征'],
			('多发性内分泌腺瘤病',): ['多发性内分泌腺瘤病'],
			('多发性?内分泌腺瘤病?(Ⅰ|1|I)型',): ['多发性内分泌腺瘤病I型'],
			('多发性?内分泌腺瘤病?2[Aa]型',): ['多发性内分泌腺瘤病2A型'],
			('高同型半胱氨酸血症',): ['高同型半胱氨酸血症'],
			('中枢性尿崩症',): ['中枢性尿崩症'],
			('心源性休克',): ['心源性休克'],
			('垂体卒中',): ['垂体卒中'],
			('真性?红细胞增多症',): ['真性红细胞增多症'],
			('β-地中海贫血',): ['β-地中海贫血'],
			('视神经脊髓炎',): ['视神经脊髓炎'],
			('卟啉病',): ['卟啉病'],
			('利什曼病',): ['利什曼病'],
			('免疫性血小板减少症',): ['免疫性血小板减少症'],
			('胰岛素瘤',): ['胰岛素瘤'],
			('血吸虫病',): ['血吸虫病'],
			('小细胞肺癌', '肺(中叶)?小细胞癌'): ['小细胞肺癌'],
			('股骨头无菌性坏死',): ['股骨头无菌性坏死'],
			('胰高血糖素瘤',): ['胰高血糖素瘤'],
			('嗜铬细胞瘤',): ['嗜铬细胞瘤'],
			('长QT综合(征|症)',): ['长QT综合征'],
			('预激综合征',): ['预激综合征'],
			('低钾性周期性麻痹',): ['低钾性周期性麻痹'],
			('桥本甲状腺炎',): ['桥本甲状腺炎'],
			('法洛四联症',): ['法洛四联症'],
			('肥厚型心肌病',): ['肥厚型心肌病'],
			('颅内生殖细胞瘤',): ['颅内生殖细胞瘤'],
			('急性淋巴细胞白血病',): ['急性淋巴细胞白血病'],
			('多发性骨髓瘤',): ['多发性骨髓瘤'],
			('肾脏淀粉样变',): ['肾脏淀粉样变'],
			('红斑型天疱疮',): ['红斑型天疱疮'],
		}

		self.CLASS_TO_DIS_CODES_SUPP = {
			'致心律失常性右室心肌病': ['OMIM:107970', 'ORPHA:217656'],
			'左室致密化不全':['CCRD:52.4', 'OMIM:604169', 'ORPHA:54260'],
			'抗LGI1抗体相关脑炎': ['CCRD:9.2', 'ORPHA:163908'],
			'生长激素缺乏症': ['ORPHA:631'], # Non-acquired isolated growth hormone deficiency;
			'CADASIL': ['CCRD:42', 'OMIM:125310', 'ORPHA:136'],
			'成人Still病': ['ORPHA:829'],
			'坏疽性脓皮病': ['ORPHA:48104'], #
			'白赛病': ['OMIM:109650', 'ORPHA:117'],
			'Carney complex': ['ORPHA:1359'],
			'干燥综合征': ['OMIM:270150', 'ORPHA:289390'],
			'Addison病': ['OMIM:240200', 'ORPHA:85138'],
			'嗜酸性粒细胞性胃肠炎': ['ORPHA:2070'],
			'慢性嗜酸性粒细胞性肺炎': ['ORPHA:2902'],
			'嗜酸性粒细胞性肺炎': ['ORPHA:182101'],
			'ANCA相关血管炎': ['ORPHA:156152'],
			'中枢性性早熟': ['ORPHA:759'],
			'Good综合征': ['ORPHA:169105'],
			'多发性内分泌腺瘤病': ['ORPHA:276161'],
			'多发性内分泌腺瘤病I型': ['ORPHA:652', 'OMIM:131100'],
			'多发性内分泌腺瘤病2A型': ['ORPHA:247698'],
			'高同型半胱氨酸血症': ['CCRD:45'],
			'中枢性尿崩症': ['ORPHA:178029'],
			'心源性休克': ['ORPHA:97292'],
			'垂体卒中': ['ORPHA:95613'],
			'真性红细胞增多症': ['OMIM:263300', 'ORPHA:729'],
			'β-地中海贫血': ['ORPHA:848', 'OMIM:613985', 'OMIM:603902'],
			'视神经脊髓炎': ['CCRD:81', 'ORPHA:71211'],
			'卟啉病': ['CCRD:92', 'ORPHA:738'],
			'利什曼病': ['ORPHA:507', 'OMIM:608207'],
			'免疫性血小板减少症': ['OMIM:188030', 'ORPHA:3002'],
			'胰岛素瘤': ['ORPHA:97279'],
			'血吸虫病': ['ORPHA:1247'],
			'小细胞肺癌': ['OMIM:182280', 'ORPHA:70573'],
			'股骨头无菌性坏死': ['OMIM:150600', 'ORPHA:2380'],
			'胰高血糖素瘤': ['ORPHA:97280'],
			'嗜铬细胞瘤': ['OMIM:171300'],
			'长QT综合征': ['CCRD:14.1', 'ORPHA:768'],
			'预激综合征': ['OMIM:194200'],
			'低钾性周期性麻痹': ['ORPHA:681', 'OMIM:170400', 'OMIM:613345'],
			'桥本甲状腺炎': ['OMIM:140300'],
			'法洛四联症': ['ORPHA:3303', 'OMIM:187500'],
			'肥厚型心肌病': [ # CARDIOMYOPATHY, FAMILIAL HYPERTROPHIC; CMH
				'OMIM:192600', # CMH1
				'OMIM:115195', # CMH2
				'OMIM:115196', # CMH3
				'OMIM:115197', # CMH4
				'OMIM:600858', # CMH6
				'OMIM:613690', # CMH7
				'OMIM:608751', # CMH8
				'OMIM:613765', # CMH9
				'OMIM:608758', # CMH10
				'OMIM:612098', # CMH11
				'OMIM:612124', # CMH12
				'OMIM:613243', # CMH13
				'OMIM:613251', # CMH14
				'OMIM:613255', # CMH15
				'OMIM:613838', # CMH16
				'OMIM:613873', # CMH17
				'OMIM:613874', # CMH18
				'OMIM:613876', # CMH20
				'OMIM:614676', # CMH21
				'OMIM:615248', # CMH22
				'OMIM:612158', # CMH23
				'OMIM:601493', # CMH24
				'OMIM:607487', # CMH25
				'OMIM:617047', # CMH26
				'OMIM:618052', # CMH27
			],
			'颅内生殖细胞瘤': ['ORPHA:91352'],
			'急性淋巴细胞白血病': ['OMIM:613065', 'ORPHA:513'],
			'多发性骨髓瘤': ['ORPHA:29073', 'OMIM:254500'],
			'肾脏淀粉样变': ['ORPHA:85450'],
			'红斑型天疱疮': ['ORPHA:79480'],
		}

		for p, cls_list in self.PATTERNS_TO_CLASSES_SUPP.items():
			dict_list_extend(p, cls_list, self.PATTERNS_TO_CLASSES)

		for cls_str, dis_codes in self.CLASS_TO_DIS_CODES_SUPP.items():
			new_dis_codes = set()
			for dis_code in dis_codes:
				rd_code = dis_code if dis_code.startswith('RD:') else self.source_to_rd.get(dis_code, '')
				if rd_code:
					rd_desc_set = get_all_descendents(rd_code, self.rd_dict)
					new_dis_codes.update([source_code for rd in rd_desc_set for source_code in self.rd_dict[rd]['SOURCE_CODES']])
				else:
					new_dis_codes.add(rd_code)
			dict_set_update(cls_str, dis_codes, self.CLASS_TO_DIS_CODES)

		self.cls_to_patterns = {} # {CLASS_STR: [PATTERNS]}
		for patterns, class_strs in self.PATTERNS_TO_CLASSES.items():
			for cls in class_strs:
				dict_list_extend(cls, patterns, self.cls_to_patterns)
		# self.cls_to_terms = self.get_rd_to_cns_list() # {RD_CODE: [term1, term2, ...]}
		self.cls_to_terms = {}


	def init_path(self, input_folder):
		self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_2000', '罕见病数据-总')
		self.DISEASE_MAP_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'disease_2000_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'disease_2000_map.csv')

		self.patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'PUMC_2000')
		os.makedirs(self.patient_save_folder, exist_ok=True)

		self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'test_pids.json')
		self.test_pids = None

		self.test_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test')


	def get_test_json(self):
		return os.path.join(self.test_patient_save_folder, 'PUMC_2000-' + self.get_json_name())


	def get_val_json(self):
		return os.path.join(self.val_patient_save_folder, 'PUMC_2000-' + self.get_json_name())


	def process(self):
		test_pids = self.get_test_pids()
		pid_to_diag_codes = self.get_pid_to_diag_codes()  # {PID: [dis_code1, ...]}
		pid_to_field_info = self.get_pid_to_field_info()
		patients = []
		for pid in test_pids:
			field_info = pid_to_field_info[pid]
			diag_codes = pid_to_diag_codes[pid]
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			hpo_codes = self.get_hpo_codes(field_info, None)
			hpo_codes = self.process_pa_hpo_list(hpo_codes, reduce=True)
			if len(hpo_codes) == 0:
				print(f'warning: empty hpos in pid {pid}')
			patients.append([hpo_codes, diag_codes])
		print('Patent Number: {}'.format(len(patients)))
		explainer = LabeledDatasetExplainer(patients)
		json.dump(explainer.explain(), open(self.get_stat_json(), 'w'), indent=2, ensure_ascii=False)
		json.dump(patients, open(self.get_test_json(), 'w'), indent=2)


	@check_load_save('test_pids', 'TEST_PIDS_JSON', JSON_FILE_FORMAT)
	def get_test_pids(self):
		self.all_dis_set = set(HPOReader().get_dis_list())
		pid_to_diag_codes = self.get_pid_to_diag_codes()
		del_pids = set(self.get_remove_pids())
		test_pids = []
		for pid, diag_codes in pid_to_diag_codes.items():
			if pid in del_pids:
				continue
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			if len(diag_codes) == 0:
				continue
			test_pids.append(pid)
		return test_pids


	def get_remove_patient_info(self, input_folder, keep_pids=None):

		keep_pids = keep_pids or {}
		json_list = get_file_list(input_folder, lambda p:p.endswith('.json'))
		info_set = set()
		for json_path in json_list:
			pid = split_path(json_path)[1]
			if pid in keep_pids:
				continue
			field_info = json.load(open(json_path))
			info_set.add((field_info['病案号']['RAW_TEXT'].strip(), field_info['姓名']['RAW_TEXT'].strip()))
		return info_set


	def get_remove_pids(self, keep=None):
		if keep is None:
			keep_pids = {}
		elif keep == TEST_DATA:
			keep_pids = json.load(open(os.path.join(DATA_PATH, 'raw', 'PUMC', 'val_test_split', 'test_pids.json')))
		elif keep == VALIDATION_DATA:
			keep_pids = json.load(open(os.path.join(DATA_PATH, 'raw', 'PUMC', 'val_test_split', 'val_pids.json')))
		else:
			assert False
		remove_info = self.get_remove_patient_info(
			os.path.join(DATA_PATH, 'raw', 'PUMC', 'case87-doc-hy-strict-enhance'), keep_pids=keep_pids)
		pid_to_field_info = self.get_pid_to_field_info()
		del_pids = set()
		for pid, field_info in pid_to_field_info.items():
			if (field_info['病案号']['RAW_TEXT'].strip(), field_info['姓名']['RAW_TEXT'].strip()) in remove_info:
				del_pids.add(pid)
		return del_pids


	def get_pid_to_diag_codes(self):
		df = pd.read_csv(self.DISEASE_MAP_AUTO_CSV, index_col=0)
		pid_to_diag_codes = {}
		for pid, row_series in df.iterrows():
			pid = '{:04}'.format(pid)
			diag_codes = eval(row_series['DIAG_CODES'])
			if len(diag_codes) == 0:
				continue
			pid_to_diag_codes[pid] = diag_codes
		return pid_to_diag_codes


	def get_disease_codes(self, diag_text):
		"""
		Args:
			diag_text (str)
		Returns:
			list: [dis_code, ...]
			list: [diag_class, ...]
		"""
		res_strs, _ = self.searcher.search(diag_text)
		ret_codes = set()
		for res_str in res_strs:
			if res_str.startswith('RD:'):
				rd_codes = get_all_descendents(res_str, self.rd_dict)
				for rd_code in rd_codes:
					ret_codes.update(self.rd_dict[rd_code]['SOURCE_CODES'])
			else:
				ret_codes.update(self.CLASS_TO_DIS_CODES[res_str])
		return sorted(list(ret_codes)), res_strs


	def gen_disease_auto_match_csv(self):
		pid_to_field_info = self.get_pid_to_field_info()
		row_dicts, row_dicts_no_map = [], []
		explainer = Explainer()
		for pid, filed_info in pid_to_field_info.items():
			raw_diag_text = filed_info['出院诊断']['RAW_TEXT']
			diag_texts = re.split('\s', raw_diag_text)
			diag_codes, diag_cls = set(), set()
			for diag_text in diag_texts:
				if diag_text.strip():
					codes, cls_list = self.get_disease_codes(diag_text)
					diag_codes.update(codes); diag_cls.update(cls_list)
			diag_codes, diag_cls = list(diag_codes), list(diag_cls)

			row_dict = {
				'PID': pid,
				'DIAG_TEXT': raw_diag_text,
				'CLASS': diag_cls,
				'CLASS_CNS': explainer.add_cns_info(diag_cls),
				'DIAG_CODES': diag_codes,
				'DIAG_CODES_CNS': explainer.add_cns_info(diag_codes)
			}
			if diag_codes:
				row_dicts.append(row_dict)
			else:
				row_dicts_no_map.append(row_dict)

		row_dicts.extend(row_dicts_no_map)
		pd.DataFrame(row_dicts).to_csv(self.DISEASE_MAP_AUTO_CSV, index=False,
			columns=['PID', 'DIAG_TEXT', 'CLASS', 'CLASS_CNS', 'DIAG_CODES', 'DIAG_CODES_CNS'])


	def gen_new_test_pids(self):
		"""
		"""
		import random
		pid_to_cls_list = self.get_pid_to_cls_list()
		selected_cls_list = self.get_selected_class()
		pids_with_cls_select = set()
		for pid, cls_list in pid_to_cls_list.items():
			for cls in cls_list:
				if cls in selected_cls_list:
					pids_with_cls_select.add(pid)
					break

		tmp_pid2diag_codes, pid2diag_codes = self.get_pid_to_diag_codes(), {}
		for pid, diag_codes in tmp_pid2diag_codes.items():
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			if diag_codes:
				pid2diag_codes[pid] = diag_codes
		old_test_pids_json = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'test_pids_too_many_cushing.json')
		old_test_pids = json.load(open(old_test_pids_json))
		old_test_pids = [pid for pid in old_test_pids if pid in pids_with_cls_select]
		pumc_87_test_pids = list(self.get_remove_pids(keep=VALIDATION_DATA)); print('Add pumc_87 test:', len(pumc_87_test_pids))
		old_test_pids = sorted(list(set(old_test_pids + pumc_87_test_pids))) # add pumc_87 test
		cushing_key = {'OMIM:615954', 'OMIM:614190', 'OMIM:615830', 'ORPHA:99889', 'ORPHA:189427', 'OMIM:219090', 'ORPHA:96253', 'ORPHA:553', 'ORPHA:99892', 'OMIM:610475', 'OMIM:219080', 'ORPHA:99893', 'OMIM:610489'}
		cushing_pids = [pid for pid, diag_codes in pid2diag_codes.items() if set(diag_codes) == cushing_key]
		cushing_pids = [pid for pid in cushing_pids if pid in old_test_pids]
		spe_del_pids = [pid for pid, diag_codes in pid2diag_codes.items() if set(diag_codes) == {'CCRD:16', 'OMIM:148000', 'OMIM:270150', 'ORPHA:160'}] # not to meet 911, del 1 patient
		other_pids = [pid for pid in old_test_pids if pid not in cushing_pids and pid not in spe_del_pids]
		print(len(cushing_pids), len(other_pids))   # 889 761
		sample_cushing_pids = random.sample(cushing_pids, 200)
		test_pids = sorted(other_pids + sample_cushing_pids)
		json.dump(test_pids, open(self.TEST_PIDS_JSON, 'w'), indent=2)


	def get_pid_to_cls_list(self):
		df = pd.read_csv(self.DISEASE_MAP_AUTO_CSV, index_col=0)
		pid_to_cls_list = {}
		for pid, row_series in df.iterrows():
			pid = '{:04}'.format(pid)
			cls_list = eval(row_series['CLASS'])
			if len(cls_list) == 0:
				continue
			pid_to_cls_list[pid] = cls_list
		return pid_to_cls_list


	def get_selected_class(self):
		return [
			'21-羟化酶缺乏', 'Bartter综合征', 'Brugada综合征', 'CREST综合征', 'Castleman病', 'Churg-Strauss综合征',
			'Cronkhite-Canada综合征', 'Ehlers-Danlos综合征', 'Fabry病', 'Fanconi综合征', 'Gitelman', 'IgG4相关性疾病',
			'McCune-Albright综合征', 'POEMS综合症', 'SAPHO', '先天性脊柱侧凸', '原发性系统性淀粉样变性', '嗜酸性肉芽肿血管炎',
			'库欣综合征', '心脏离子通道病（长QT间期综合征）', '扩张型心肌病', '抗NMDAR受体脑炎', '朗格汉斯细胞组织细胞增生症',
			'烟雾病', '特发性肺动脉高压', '特纳综合征', '皮肤恶性黑色素瘤', '直肠神经内分泌瘤', '着色性干皮病', '系统性硬化症',
			'线粒体脑肌病', '结节性硬化症', '肝豆状核变性', '肺泡蛋白沉积症', '致心律失常性右室心肌病', '蕈样肉芽肿', '重症肌无力', '限制性心肌病'
		]


class PUMC2000BalanceGenerator(PUMC2000PatientGenerator):
	def __init__(self, fields=None, del_diag_hpo=False, mark='', input_folder=None, hpo_reader=HPOReader()):
		super(PUMC2000BalanceGenerator, self).__init__(fields, del_diag_hpo, mark, input_folder, hpo_reader)

	def init_path(self, input_folder):
		self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_2000', '罕见病数据-总')
		self.DISEASE_MAP_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'disease_2000_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(DATA_PATH, 'raw', 'PUMC_2000', 'disease_2000_map.csv')

		self.patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'PUMC_2000_BALANCE')
		os.makedirs(self.patient_save_folder, exist_ok=True)

		self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_2000_BALANCE', 'test_pids.json')
		self.test_pids = None

		self.test_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test')

	def get_test_json(self):
		return os.path.join(self.test_patient_save_folder, 'PUMC_2000_BALANCE-' + self.get_json_name())

	def get_val_json(self):
		return os.path.join(self.val_patient_save_folder, 'PUMC_2000_BALANCE-' + self.get_json_name())


	@check_load_save('test_pids', 'TEST_PIDS_JSON', JSON_FILE_FORMAT)
	def get_test_pids(self):
		import random
		def make_key(cls_list):
			return tuple(sorted(cls_list))
		max_sample_num = 10

		# selected if diag_codes is not empty
		self.all_dis_set = set(HPOReader().get_dis_list())
		pid_to_diag_codes = self.get_pid_to_diag_codes()
		test_pids, pid_to_slice_diag_codes = [], {}
		for pid, diag_codes in list(pid_to_diag_codes.items()):
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			if len(diag_codes) == 0:
				continue
			pid_to_slice_diag_codes[pid] = diag_codes
			test_pids.append(pid)

		# selected specified diag class
		pid_to_cls_list = self.get_pid_to_cls_list()
		selected_cls_list = self.get_selected_class()
		pids_with_cls_select = set()
		for pid, cls_list in pid_to_cls_list.items():
			if len(cls_list) > 1:
				continue
			for cls in cls_list:
				if cls in selected_cls_list:
					pids_with_cls_select.add(pid)
					break
		test_pids = [pid for pid in test_pids if pid in pids_with_cls_select]

		# downsample to max_sample_num
		diag2pids = {}
		for pid in test_pids:
			dict_list_add(make_key(pid_to_slice_diag_codes[pid]), pid, diag2pids)
		test_pids = []
		for pids in diag2pids.values():
			test_pids.extend(random.sample(pids, min(len(pids), max_sample_num)))

		return test_pids


if __name__ == '__main__':


	# ==============================================================
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	base_folder = '/home/yhuang/RareDisease/bert_syn_project/data/preprocess/pumc_2000'

	paras = [

		('DictBert-AlbertDDML', os.path.join(base_folder, 'dict_bert-albertTinyDDMLSim')),

	]

	# 961 patients
	fields = ['现病史', '入院情况', '出院诊断', '既往史']
	for mark, input_folder in paras:
		pg = PUMC2000PatientGenerator(fields=fields, del_diag_hpo=False, mark=mark, input_folder=input_folder, hpo_reader=hpo_reader)
		pg.process()


