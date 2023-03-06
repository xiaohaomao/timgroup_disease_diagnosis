
import re
import os
import json
from sklearn.model_selection import train_test_split

from core.utils.utils import get_file_list, get_logger, slice_list_with_keep_set, dict_set_update
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.patient.patient_generator import PatientGenerator
from core.patient.PUMC_OMIM_DICT import CLASS_TO_OMIM
from core.patient.PUMC_ORPHA_DICT import CLASS_TO_ORPHANET
from core.patient.PUMC_CCRD_DICT import CLASS_TO_CCRD
from core.explainer.dataset_explainer import LabeledDatasetExplainer
from core.reader import HPOReader, HPOFilterDatasetReader


class PUMCPatientGenerator(PatientGenerator):
	def __init__(self, fields=None, del_diag_hpo=False, mark='', input_folder=None, hpo_reader=HPOReader()):
		super(PUMCPatientGenerator, self).__init__(hpo_reader)
		self.fields = fields or ['现病史', '既往史', '入院情况', '出院诊断']
		self.del_diag_hpo = del_diag_hpo
		self.mark = mark
		self.init_path(input_folder)
		self.init_pattern_dict()

		self.TEST_SIZE = 0.6
		self.MIN_HPO_NUM = 3
		self.check_diag_to_class()


	def init_pattern_dict(self):
		self.PATTERNS_TO_CLASSES = {
			('Bartter综合(征|症)?',):['Bartter综合征'],
			('Fanconi',):['Fanconi综合征'],
			('Brugada',):['Brugada综合征'],
			('[Cc]astleman.{0,3}(病|综合征|综合症)?',): ['Castleman病'],  # 由于HPO只提供了castleman病的一种亚型并发症（OMIM:148000，卡波西肉瘤），故删去
			('(POEMS|POMES)综合(征|症)', '(POEMS|POMES)病',):['POEMS综合症'],
			('Castleman病.*?POEMS变异型',): ['Castleman病', 'POEMS综合症'],
			('[Cc]ronkhite.{0,3}[Cc]anad[a|e]',): ['Cronkhite-Canada综合征'],
			('Ehlers.{0,3}Danlos(综合征)?',):['Ehlers-Danlos综合征'],
			('Fabry病', '法布(里|雷)病', ):['Fabry病'],
			('Gitelman', 'Gitleman', 'Giltelman'):['Gitelman'],
			('IgG4相关性疾病', 'IgG4相关性淋巴结病', 'IgG4相关淋巴结病'):['IgG4相关性疾病'],  # 由于HPO只提供了OMIM:228800（IGG4相关的纤维化疾病）这一种，故删去
			('McCune-Albright综合征',):['McCune-Albright综合征'],
			('SAPHO',):['SAPHO'],
			('恶性黑色素瘤', '黑色素瘤'):['皮肤恶性黑色素瘤'],
			('肺泡蛋白沉积症',):['肺泡蛋白沉积症'],
			('(特发性|原发性)肺动脉高压', '肺动脉高压(（.*）)\s* 特发性'):['特发性肺动脉高压'],
			('库(欣|兴)氏?(综合)?(症|征|病)', '[Cc]ushing?', 'CUSHING', '皮质醇(增多)?症', '异位ACTH综合(征|症)', 'AIMAH', '非?ACTH非?依赖(性|型)(双侧)?(肾上腺(皮质)?大结节(样|性)|大结节(样|性)肾上腺(皮质)?)增生', '垂体(ACTH|促肾上腺皮质激素)(型|性)?(微|大)?腺瘤', '库欣腺瘤'): ['库欣综合征'],
			('肝豆状核变性',):['肝豆状核变性'],
			('结节性硬化症',):['结节性硬化症'],
			('抗.{0,3}NMDAR?(抗体)?(受体)?.{0,3}(相关)?脑炎',): ['抗NMDAR受体脑炎'],  # 注：HPO未提供疾病注释，故自动删去
			('扩张(型|性)心肌病',):['扩张型心肌病'],
			('朗格汉斯细胞组织细胞增生症',):['朗格汉斯细胞组织细胞增生症'],  # 注: 没有提供'OMIM:604856', 'ORPHA:389'的疾病注释，是否删去待定
			('离子通道病.*?长QT间期综合征', '离子通道病'):['心脏离子通道病（长QT间期综合征）'],
			('21-?羟化酶缺乏(症|征)?', '羟化酶缺乏'):['21-羟化酶缺乏'],
			('[Cc]hurg.{0,2}[Ss]trauss(综合征)?',):['Churg-Strauss综合征'],
			('变应性?嗜酸性?肉芽肿', '(变应|嗜酸)性?粒?(细胞)?性?肉芽肿性?伴?多?血管炎',):['嗜酸性肉芽肿血管炎'],
			('(特纳|Turner)综合征',):['特纳综合征'],
			('先天性脊柱侧后?(凸|弯)',):['先天性脊柱侧凸'],
			('限制(性|型)心肌病',):['限制性心肌病'],
			('线粒体病',):['线粒体脑肌病'],
			('蕈样肉芽肿',):['蕈样肉芽肿'],
			('烟雾病', '(烟雾|Moyamoya|moyamoya|MOYAMOYA)综合征'):['烟雾病'],
			('(原发性)?(系统性|轻链型)淀粉样变性?',):['原发性系统性淀粉样变性'],
			('直肠(肿物|占位)?\s*神经内分泌(细胞)?(肿瘤|瘤|癌)', '直肠类癌'):['直肠神经内分泌瘤'],
			('致心律失常(性|型)?右室心肌病',):['致心律失常性右室心肌病'],
			('重症肌无力',):['重症肌无力'],
			('CREST综合征',):['CREST综合征'],
			('系统性硬化症',):['系统性硬化症'],
			('着色性干皮病',):['着色性干皮病'],
		}
		assert set(CLASS_TO_OMIM.keys()) == set(CLASS_TO_ORPHANET.keys()) and set(CLASS_TO_OMIM.keys()) == set(CLASS_TO_CCRD.keys())
		self.CLASS_TO_DIS_CODES = self.combine_class_to_dis_codes(CLASS_TO_OMIM, CLASS_TO_ORPHANET, CLASS_TO_CCRD)

		self.DIAG_CLASS_TO_DELETE_HPO = {
			'Fanconi综合征':['HP:0001994'],  # 范可尼综合征
			'恶性黑色素瘤':['HP:0002861', 'HP:0012056'],  # 恶性黑素瘤; 皮肤黑色素瘤
			'肺泡蛋白沉积症':['HP:0006517'],  # 肺泡蛋白沉积症
			'扩张型心肌病':['HP:0001644'],  # 扩张型心肌病
			'限制性心肌病':['HP:0001723'],  # 限制性心肌病
			'先天性脊柱侧凸':['HP:0008453'],  # 先天性脊柱后侧凸畸形
			'线粒体脑肌病':['HP:0003737'],  # 线粒体肌病
		}


	def combine_class_to_dis_codes(self, *args):
		ret_dict = {}
		for class_to_dis_codes in args:
			for class_str, dis_codes in class_to_dis_codes.items():
				dict_set_update(class_str, dis_codes, ret_dict)
		return ret_dict


	def init_path(self, input_folder):
		self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC', 'case87-doc-hy-strict-enhance')
		self.patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'PUMC')
		os.makedirs(self.patient_save_folder, exist_ok=True)
		self.test_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test')
		os.makedirs(self.test_patient_save_folder, exist_ok=True)
		self.val_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'validation')
		os.makedirs(self.val_patient_save_folder, exist_ok=True)
		self.raw_pumc_folder = os.path.join(DATA_PATH, 'raw', 'PUMC', 'val_test_split')
		os.makedirs(self.raw_pumc_folder, exist_ok=True)
		self.VAL_PIDS_JSON = os.path.join(self.raw_pumc_folder, 'val_pids.json')
		self.TEST_PIDS_JSON = os.path.join(self.raw_pumc_folder, 'test_pids.json')


	def check_diag_to_class(self):
		assert set(CLASS_TO_OMIM.keys()) == set(CLASS_TO_ORPHANET.keys())


	def get_json_name(self):
		s = f'{self.mark}-' if self.mark else ''
		return s + f'{sorted(self.fields)}-del_diag_hpo_{self.del_diag_hpo}.json'

	def get_val_pids_json(self):
		return os.path.join(self.patient_save_folder, 'val-pids-' + self.get_json_name())

	def get_val_json(self):
		return os.path.join(self.val_patient_save_folder, 'PUMC-' + self.get_json_name())

	def get_test_pids_json(self):
		return os.path.join(self.patient_save_folder, 'test-pids-' + self.get_json_name())

	def get_test_json(self):
		return os.path.join(self.test_patient_save_folder, 'PUMC-' + self.get_json_name())

	def get_del_pids_json(self):
		return os.path.join(self.patient_save_folder, 'del-pids-' + self.get_json_name())

	def get_stat_json(self):
		return os.path.join(self.patient_save_folder, 'stats-' + self.get_json_name())

	def get_pid(self, path):
		folder, file_name = os.path.split(path)
		return os.path.splitext(file_name)[0]


	def get_all_pids(self):
		return sorted(list(self.get_pid_to_field_info().keys()))


	def split_test_val(self):
		pids = self.get_all_pids()
		val_pids, test_pids = train_test_split(pids, test_size=self.TEST_SIZE)
		json.dump(val_pids, open(self.VAL_PIDS_JSON, 'w'), indent=2)
		json.dump(test_pids, open(self.TEST_PIDS_JSON, 'w'), indent=2)


	def get_test_pids(self):
		return json.load(open(self.TEST_PIDS_JSON))


	def get_val_pids(self):
		return json.load(open(self.VAL_PIDS_JSON))


	def process(self):
		pid_to_field_info = self.get_pid_to_field_info()

		pid_to_patient = {}
		for pid, field_info in pid_to_field_info.items():
			diag_codes, diag_classes = self.get_disease_codes(field_info['出院诊断']['RAW_TEXT'] + '\n' + field_info['mark']['RAW_TEXT'])
			print('PID = {}; Mark = {}; Diag Class = {}'.format(pid, field_info['mark']['RAW_TEXT'].strip(), diag_classes))
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			hpo_codes = self.get_hpo_codes(field_info, diag_classes)
			hpo_codes = self.process_pa_hpo_list(hpo_codes, reduce=True)
			patient = [hpo_codes, diag_codes]
			pid_to_patient[pid] = patient

		val_pids, test_pids = self.get_val_pids(), self.get_test_pids()
		# FIXME
		# test_pids = test_pids + val_pids

		new_val_pids = sorted([pid for pid in val_pids if len(pid_to_patient[pid][0]) >= self.MIN_HPO_NUM and len(pid_to_patient[pid][1]) > 0])
		new_test_pids = sorted([pid for pid in test_pids if len(pid_to_patient[pid][0]) >= self.MIN_HPO_NUM and len(pid_to_patient[pid][1]) > 0])
		del_pids = sorted(list(set(val_pids + test_pids) - set(new_val_pids + new_test_pids)))
		json.dump(new_val_pids, open(self.get_val_pids_json(), 'w'), indent=2)
		json.dump(new_test_pids, open(self.get_test_pids_json(), 'w'), indent=2)
		json.dump(del_pids, open(self.get_del_pids_json(), 'w'), indent=2)

		all_patients = [pid_to_patient[pid] for pid in set(new_val_pids + new_test_pids)]
		explainer = LabeledDatasetExplainer(all_patients)
		json.dump(explainer.explain(), open(self.get_stat_json(), 'w'), indent=2, ensure_ascii=False)

		json.dump([pid_to_patient[pid] for pid in new_val_pids], open(self.get_val_json(), 'w'), indent=2)
		json.dump([pid_to_patient[pid] for pid in new_test_pids], open(self.get_test_json(), 'w'), indent=2)


	def process_machine_tag(self):
		tmp_mark = self.mark; self.mark = ''
		val_pids, test_pids = json.load(open(self.get_val_pids_json())), json.load(open(self.get_test_pids_json()))
		raw_val_patients, raw_test_patients = json.load(open(self.get_val_json())), json.load(open(self.get_test_json()))
		pid_to_diag_codes = {}  # {PID: [dis_code1, ...]}
		pid_to_diag_codes.update({pid: patient[1] for pid, patient in zip(val_pids, raw_val_patients)})
		pid_to_diag_codes.update({pid: patient[1] for pid, patient in zip(test_pids, raw_test_patients)})
		self.mark = tmp_mark
		pid_to_field_info = self.get_pid_to_field_info()
		pid_to_patient = {}    # {PID: [hpo_code1, ...]}
		for pid in pid_to_diag_codes:
			field_info = pid_to_field_info[pid]
			hpo_codes = self.get_hpo_codes(field_info, None)    # FIXME: del_hpo
			hpo_codes = self.process_pa_hpo_list(hpo_codes, reduce=True)
			pid_to_patient[pid] = [hpo_codes, pid_to_diag_codes[pid]]

		all_patients = [pid_to_patient[pid] for pid in set(val_pids + test_pids)]
		explainer = LabeledDatasetExplainer(all_patients)
		json.dump(explainer.explain(), open(self.get_stat_json(), 'w'), indent=2, ensure_ascii=False)

		json.dump([pid_to_patient[pid] for pid in val_pids], open(self.get_val_json(), 'w'), indent=2)
		json.dump([pid_to_patient[pid] for pid in test_pids], open(self.get_test_json(), 'w'), indent=2)


	def get_labels_set_with_all_eq_sources(self, sources):
		"""
		Returns:
			set: {sorted_dis_codes_tuple, ...}; sorted_dis_codes_tuple = (dis_code1, dis_code2, ...)
		"""
		patients = json.load(open(self.get_val_json())) + json.load(open(self.get_test_json()))
		return set([tuple(sorted(dis_codes)) for hpo_codes, dis_codes in patients if self.diseases_from_all_sources(dis_codes, sources)])


	def get_pid_to_field_info(self):
		"""
		Returns:
			dict: {
				pid: {
					FIELD: {
						'RAW_TEXT': str,
						'ENTITY_LIST': [
							{
								'SPAN_LIST': [(start_pos, end_pos), ...],
								'SPAN_TEXT': str
								'HPO_CODE': str,
								'HPO_TEXT': str,
								'TAG_TYPE': str
							},
							...
						]
					}
				}
			}
		"""
		json_list = sorted(get_file_list(self.input_folder, lambda p: p.endswith('.json')))
		return {self.get_pid(json_path): json.load(open(json_path)) for json_path in json_list}


	def get_disease_codes(self, diag_text):
		"""
		Args:
			diag_text (str)
		Returns:
			list: [dis_code, ...]
			list: [diag_class, ...]
		"""
		lines = [line.strip() for line in diag_text.splitlines()]
		ret_diag_classes = []
		for line in lines:
			if len(line) == 0: continue
			for patterns, diag_classes in self.PATTERNS_TO_CLASSES.items():
				for pattern in patterns:
					# print('Line = {}; Pattern = {}; Match = {}'.format(line, pattern, re.search(re.escape(pattern), line)))
					if re.search(pattern, line):
						ret_diag_classes.extend(diag_classes)
						break
		dis_codes = set()
		for dc in ret_diag_classes:
			dis_codes.update(self.CLASS_TO_DIS_CODES[dc])
		return list(dis_codes), list(set(ret_diag_classes))


	def get_hpo_codes(self, field_info, diag_classes):
		"""
		Args:
			field_info (dict)
			diag_class (list)
		Returns:
			list: [HPOCode, ...]
		"""
		hpo_codes = []
		for field_name, tag_dict in field_info.items():
			if field_name in self.fields:
				hpo_codes.extend([tag_item['HPO_CODE'] for tag_item in tag_dict['ENTITY_LIST'] if len(tag_item['HPO_CODE'].strip()) > 0])
		if self.del_diag_hpo:
			del_hpo_set = set([hpo_code for diag_class in diag_classes for hpo_code in self.DIAG_CLASS_TO_DELETE_HPO.get(diag_class, [])])
			hpo_codes = [hpo_code for hpo_code in hpo_codes if hpo_code not in del_hpo_set]
		return list(set(hpo_codes))


	def print_all_class(self):
		all_classes = set()
		for cls_list in self.PATTERNS_TO_CLASSES.values():
			all_classes.update(cls_list)
		print(sorted(list(all_classes)))



if __name__ == '__main__':
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])

	# generate machine tag patients ========================
	base_folder = '/home/yhuang/RareDisease/bert_syn_project/data/preprocess/pumc_87'

	paras = [

		('DictBert-AlbertDDML', os.path.join(base_folder, 'dict_bert-albertTinyDDMLSim')),
	]


	fields = ['现病史', '入院情况', '出院诊断', '既往史']
	for mark, input_folder in paras:
		pg = PUMCPatientGenerator(fields=fields, del_diag_hpo=False, mark=mark, input_folder=input_folder, hpo_reader=hpo_reader)
		pg.process_machine_tag()
