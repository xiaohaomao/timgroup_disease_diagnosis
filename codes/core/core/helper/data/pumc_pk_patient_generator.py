

import os
import pandas as pd
import json
import re

from core.patient.patient_generator import PatientGenerator
from core.patient.PUMC_OMIM_DICT import CLASS_TO_OMIM
from core.patient.PUMC_ORPHA_DICT import CLASS_TO_ORPHANET
from core.patient.PUMC_CCRD_DICT import CLASS_TO_CCRD
from core.reader import HPOReader, HPOFilterDatasetReader
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, get_file_list, dict_set_update, slice_list_with_keep_set
from core.explainer import Explainer, LabeledDatasetExplainer


class PUMCPKPatientGenerator(PatientGenerator):
	def __init__(self, fields=None, mark='', input_folder=None, hpo_reader=HPOReader()):
		super(PUMCPKPatientGenerator, self).__init__(hpo_reader=hpo_reader)
		self.ALL_FIELDS = ['主诉', '现病史', '体格检查', '辅助检查', '专科情况', '入院诊断', '既往史', '家族史', '月经史', '个人史', '婚育史']
		self.fields = fields or self.ALL_FIELDS
		self.mark = mark
		self.init_path(input_folder)

		self.MASK_PATTERN = re.compile('\[(DIS_MASK|GENE_MASK|DRUG_MASK)\]')
		assert set(CLASS_TO_OMIM.keys()) == set(CLASS_TO_ORPHANET.keys()) and set(CLASS_TO_OMIM.keys()) == set(CLASS_TO_CCRD.keys())
		self.CLASS_TO_DIS_CODES = self.combine_class_to_dis_codes(CLASS_TO_OMIM, CLASS_TO_ORPHANET, CLASS_TO_CCRD)
		self.PATTERNS_TO_CLASSES = self.gen_patterns_to_classes()


	def gen_patterns_to_classes(self):
		return {
			# 儿科
			('Prader-Willi综合征',): ['Prader-Willi综合征'],
			('肝豆状核变性',): ['肝豆状核变性'],
			('McCune-Albright综合征',): ['McCune-Albright综合征'],

			# 心内科、心外科、骨科
			('马方综合征',): ['马方综合征'],
			('致心律失常性右室心肌病',): ['致心律失常性右室心肌病'],
			('Brugada综合征',): ['Brugada综合征'],
			('限制性心肌病',): ['限制性心肌病'],

			# 神经科
			('肌萎缩侧索硬化',): ['肌萎缩侧索硬化'],
			('全身型重症肌无力',): ['重症肌无力'],
			('多系统萎缩',): ['多系统萎缩'],

			# 肾内科
			('Alport综合征',): ['Alport综合征'],
			('法布雷病',): ['Fabry病'],
			('Gitelman综合征',): ['Gitelman'],

			# 血液科
			('阵发性睡眠性血红蛋白尿',): ['阵发性睡眠性血红蛋白尿'],
			('POEMS综合征',): ['POEMS综合症'],
			('尼曼匹克病',): ['尼曼匹克病'],
		}


	def combine_class_to_dis_codes(self, *args):
		ret_dict = {}
		for class_to_dis_codes in args:
			for class_str, dis_codes in class_to_dis_codes.items():
				dict_set_update(class_str, dis_codes, ret_dict)
		return ret_dict


	def init_path(self, input_folder):
		self.raw_input_folder = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', '罕见病筛查_抹除版_0911')
		self.patient_info_xlsx = os.path.join(self.raw_input_folder, '罕见病病案号_PID.xlsx')
		self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pumc_pk')
		self.pid_diag_codes_csv = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pid_diag_codes.csv')
		self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'test_pids.json')
		self.test_pids = None

		self.patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'PUMC_PK')
		os.makedirs(self.patient_save_folder, exist_ok=True)
		self.test_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test')


	def get_patient_info_list(self):
		"""
		Returns:
			list: [{'PID': str, 'NAME': str, 'DIAG': str, 'CASE_ID': str, 'DEPARTMENT': str}]
		"""
		df = pd.read_excel(self.patient_info_xlsx, dtype=str)
		ret_list = []
		for idx, row in df.iterrows():
			ret_list.append({
				'PID': row['PID'].strip(),
				'NAME': row['姓名'].strip(),
				'DIAG': row['诊断'].strip(),
				'CASE_ID': row['病案号'].strip(),
				'DEPARTMENT': row['科室'].strip()
			})
		return ret_list


	def preprocess_raw(self):
		def replace_mask(input_dict):
			for field, info in input_dict.items():
				if info['RAW_TEXT'].find('DIS_MASK') > -1: assert info['RAW_TEXT'].find('[DIS_MASK]') > -1
				if info['RAW_TEXT'].find('GENE_MASK') > -1: assert info['RAW_TEXT'].find('[GENE_MASK]') > -1
				if info['RAW_TEXT'].find('DRUG_MASK') > -1: assert info['RAW_TEXT'].find('[DRUG_MASK]') > -1
				info['RAW_TEXT'] = self.MASK_PATTERN.sub('', info['RAW_TEXT'])
		def combine_json(name, input_jsons, output_json):
			input_dict_list = [json.load(open(p)) for p in input_jsons]
			for input_dict in input_dict_list:
				replace_mask(input_dict)
			if len(input_dict_list) == 1:
				json.dump(input_dict_list[0], open(output_json, 'w'), indent=2, ensure_ascii=False)
			else:
				all_fields = set(self.ALL_FIELDS + ['姓名'])
				for input_dict in input_dict_list:
					assert set(input_dict.keys()) == all_fields
				if name != input_dict_list[0]['姓名']['RAW_TEXT']:
					print('name = {}; name in json = {}'.format(name, input_dict_list[0]['姓名']['RAW_TEXT']))	# 王璟鑫、崔雪晨、王国友
					assert False
				new_dict = {'姓名': {'RAW_TEXT': name, 'ENTITY_LIST': []}}

				for field in self.ALL_FIELDS:
					new_dict[field] = {
						'RAW_TEXT': '\n'.join([d[field]['RAW_TEXT'].strip() for d in input_dict_list]),
						'ENTITY_LIST': []
					}
				json.dump(new_dict, open(output_json, 'w'), indent=2, ensure_ascii=False)
		patient_info_list = self.get_patient_info_list()
		name2pid = {d['NAME']: d['PID'] for d in patient_info_list}
		json_list = get_file_list(self.raw_input_folder, lambda p: p.endswith('.json'))
		output_folder = self.input_folder
		os.makedirs(output_folder, exist_ok=True)
		for name, pid in name2pid.items():
			pa_jsons = [json_path for json_path in json_list if json_path.find(name) > -1]
			output_json = os.path.join(output_folder, f'{pid}.json')
			print(f'{pa_jsons} -> {output_json}')
			combine_json(name, pa_jsons, output_json)


	@check_load_save('test_pids', 'TEST_PIDS_JSON', JSON_FILE_FORMAT)
	def get_test_pids(self):
		patient_info_list = self.get_patient_info_list()
		return sorted([d['PID'] for d in patient_info_list])


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
		all_pids = self.get_test_pids()
		return {pid: json.load(open(os.path.join(self.input_folder, f'{pid}.json'))) for pid in all_pids}


	def get_disease_codes(self, diag_text):
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


	def get_pid_to_dis_codes(self):
		patient_info_list = self.get_patient_info_list()
		return {d['PID']: self.get_disease_codes(d['DIAG'])[0] for d in patient_info_list}


	def show_pid_diag_csv(self):
		patient_info_list = self.get_patient_info_list()
		explainer = Explainer()
		df_dict = []
		for patient_info in patient_info_list:
			diag_text = patient_info['DIAG']
			dis_codes, dis_classes = self.get_disease_codes(diag_text)
			df_dict.append({
				'PID': patient_info['PID'],
				'NAME': patient_info['NAME'],
				'DIAG_TEXT': diag_text,
				'DIS_CLASSES': dis_classes,
				'DIS_CODES': dis_codes,
				'DIS_CODES_CNS': explainer.add_cns_info(dis_codes)
			})
		pd.DataFrame(df_dict).to_csv(self.pid_diag_codes_csv, index=False, columns=['PID', 'NAME', 'DIAG_TEXT', 'DIS_CLASSES', 'DIS_CODES', 'DIS_CODES_CNS'])


	def get_hpo_codes(self, field_info):
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
		return list(set(hpo_codes))


	def get_json_name(self):
		s = f'{self.mark}-' if self.mark else ''
		return s + f'{sorted(self.fields)}.json'


	def get_stat_json(self):
		return os.path.join(self.patient_save_folder, 'stats-' + self.get_json_name())


	def get_test_json(self):
		return os.path.join(self.test_patient_save_folder, 'PUMC_PK-' + self.get_json_name())


	def process(self):
		test_pids = self.get_test_pids()
		pid_to_diag_codes = self.get_pid_to_dis_codes()  # {PID: [dis_code1, ...]}
		pid_to_field_info = self.get_pid_to_field_info()
		patients = []
		for pid in test_pids:
			field_info = pid_to_field_info[pid]
			diag_codes = pid_to_diag_codes[pid]
			diag_codes = slice_list_with_keep_set(diag_codes, self.all_dis_set)
			hpo_codes = self.get_hpo_codes(field_info)
			hpo_codes = self.process_pa_hpo_list(hpo_codes, reduce=True)
			if len(hpo_codes) == 0:
				print(f'warning: empty hpos in pid {pid}')
			patients.append([hpo_codes, diag_codes])
		print('Patent Number: {}'.format(len(patients)))
		explainer = LabeledDatasetExplainer(patients)
		json.dump(explainer.explain(), open(self.get_stat_json(), 'w'), indent=2, ensure_ascii=False)
		json.dump(patients, open(self.get_test_json(), 'w'), indent=2)


if __name__ == '__main__':


	base_folder = '/home/xhmao19/project/hy_works/2020_10_20_RareDisease-master/bert_syn_project/data/preprocess/'




	score_thres_list = ['2020_11_13_tagged_unlabel_pumc_pk']
	for i_paras in score_thres_list:

		outfolder_name = ''
		outfolder_name += i_paras
		print(outfolder_name)

		paras = [(outfolder_name, os.path.join(base_folder, 'pumc_pk/' + outfolder_name))]


		fields = ['主诉', '现病史', '体格检查', '辅助检查', '专科情况', '入院诊断', '既往史']
		hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])

		for mark, input_folder in paras:
			pg = PUMCPKPatientGenerator(fields=fields, mark=mark, input_folder=input_folder, hpo_reader=hpo_reader)
			pg.process()


