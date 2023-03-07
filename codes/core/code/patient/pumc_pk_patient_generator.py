
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

		#
		# self.ALL_FIELDS = ['零、题目','一、专家导读','二、病例介绍-患者','二、病例介绍-主诉','二、病例介绍-现病史','二、病例介绍-既往史',
        #            '二、病例介绍-个人史','二、病例介绍-月经婚育史','二、病例介绍-家族史','二、病例介绍-查体','二、病例介绍-辅助检查',
        #            '二、病例介绍-初步诊断','二、病例介绍-手术治疗','三、主治医师总结病史并提出会诊目的-总结病史',
        #            '三、主治医师总结病史并提出会诊目的-目前诊断','五、结局及转归','六、专家点评']






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
			('致心律失常性右室发育不良/心肌病',):['致心律失常性右室心肌病'],
			('Brugada综合征',): ['Brugada综合征'],
			('限制性心肌病',): ['限制性心肌病'],
			('限制型心肌病',):['限制性心肌病'],

			# 神经科
			('肌萎缩侧索硬化',): ['肌萎缩侧索硬化'],
			('全身型重症肌无力',): ['重症肌无力'],
			('多系统萎缩',): ['多系统萎缩'],

			# 肾内科
			('Alport综合征',): ['Alport综合征'],
			('法布雷病',): ['Fabry病'],
			('Gitelman综合征',): ['Gitelman'],

			## 血液科
			('阵发性睡眠性血红蛋白尿',): ['阵发性睡眠性血红蛋白尿'],
			('POEMS综合征',): ['POEMS综合症'],
			('尼曼匹克病',): ['尼曼匹克病'],
			('阴道斜隔综合征',): ['阴道闭锁'],  # 百度上查有三类 但是我们的知识库里面只有两类

			# 2020 11 22 MDT 疾病的code 编码
			### todo 2020 11 29 非常重要, 现在要删去几个 找不到疾病编码的 有#！*！标记的就是 ###

			#
			('阴道斜隔综合征',): ['阴道闭锁'],# 百度上查有三类 但是我们的知识库里面只有两类
			('杂合子型家族性高胆固醇血症',): ['家族性高胆固醇血症'],
			('家族性高胆固醇血症',): ['家族性高胆固醇血症'],
			('CREST综合征',): ['CREST综合征'],
			('纯合子型家族性高胆固醇血症',): ['纯合子家族性高胆固醇血症'],
			('蛋白酶体相关的自身炎症性疾病',): ['PROTEASOME-ASSOCIATED AUTOINFLAMMATORY SYNDROME 2'],
			#('联合免疫缺陷疾病中的活化PI3K-δ综合征',): ['Activated PI3K-delta syndrome'],  #！*！ rank median 从 [20,61]
			('原发性中枢神经系统血管炎',): ['Primary angiitis of the central nervous system'],
			('黏多糖贮积症II型',): ['粘多糖病第Ⅱ型'],
			('戈谢病',): ['戈谢病'],
			#('脊髓小脑性共济失调',): ['脊髓小脑性共济失调'],
			#('颅骨筋膜炎',):['Nodular fasciitis'],  #！*！  rank median 从 [149,5240]
			('大动脉炎',):['高安动脉炎,无脉病,大动脉炎'],
			('Aicardi-Goutieres综合征',):['Aicardi Goutieres综合征'],
			('SIFD',):['缺铁性红细胞贫血伴b细胞免疫缺乏，周期性发烧和发育迟缓'],
			#('Castleman病',):['Castleman病'],

			('CLOVES综合征',): ['先天性脂肪瘤样增生，血管畸形和表皮痣综合征'],      ### CONGENITAL LIPOMATOUS OVERGROWTH, VASCULAR MALFORMATIONS, AND EPIDERMAL NEVI, 致病基因 PIK3CA
			('先天性多发关节挛缩综合征',): ['先天性多发关节挛缩综合征'],  ### 目前只找五个
			#('巨趾症',): ['macrodactyly'],  #！*！  rank median 从 [800,3814]
			('McCune-Albright 综合征',): ['McCune-Albright综合征'],
			('肾上腺皮质癌',): ['肾上腺皮质癌'],
			('遗传性肾上腺皮质癌',): ['肾上腺皮质癌'],
			#('肿瘤性骨软化症',): ['Oncogenic osteomalacia'],  #！*！ rank median 从 [2926,4584]
			#('肌萎缩侧索硬化',): ['肌萎缩侧索硬化'],
			('Fabry病',): ['Fabry病'],
			('卟啉病',): ['红细胞生成性原卟啉病'],
			('Waardenburg综合征',): ['瓦登伯格综合征'],
			('SAPHO综合征',): ['SAPHO'],
			#('APDS',): ['Activated PI3K-delta syndrome'], #！*！ rank median 从 [20,61]
			('结节性硬化症',): ['结节性硬化症'],
			('原发性肥厚性骨关节病',): ['Primary hypertrophic osteoarthropathy'],
			('脊髓性肌萎缩症',): ['脊髓性肌萎缩症'],
			('蕈样肉芽肿',): ['蕈样肉芽肿'],

			('扩张型心肌病',): ['扩张型心肌病'],


			('肥厚型心肌病',):['肥厚型心肌病'],
			('Sack-Barabas综合征',):['Sack-Barabas综合征'],
			('杂合子型家族性高胆固醇血症',):['纯合子家族性高胆固醇血症'],






		}


	def combine_class_to_dis_codes(self, *args):
		ret_dict = {}
		for class_to_dis_codes in args:
			for class_str, dis_codes in class_to_dis_codes.items():
				dict_set_update(class_str, dis_codes, ret_dict)
		return ret_dict


	def init_path(self, input_folder):








		# self.raw_input_folder = os.path.join(DATA_PATH, 'raw', 'PUMC_PK')
		# self.patient_info_xlsx = os.path.join(self.raw_input_folder, '2021_05_13_RDs_76_patients_random_num_100_罕见病病案号.xlsx')
		# self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pumc_pk')
		# self.pid_diag_codes_csv = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', '2021_05_13_76RDs_pid_diag_codes.xlsx')
		# self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', '2021_05_13_wk_tagged_RDs_76_patients_random_num_100'
		# 																'_test_pids.json')
		# self.test_pids = None
























		# PUMCH-ADM
		# self.raw_input_folder = os.path.join(DATA_PATH, 'raw', 'PUMC_PK')
		# self.patient_info_xlsx = os.path.join(self.raw_input_folder, '罕见病病案号2020_12_14_pumc_PK_76_ALL_PID_control_repeat_100.xlsx')
		# self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pumc_pk')
		# self.pid_diag_codes_csv = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pid_diag_codes_76_p_value_testing_all_diseases_control_repeat_100.xlsx')
		# self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'test_PIDS_P_VALUE.json')
		# self.test_pids = None






		## PUMCH MDT
		# self.raw_input_folder = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', '罕见病筛查_mdt_1129_27份病例')
		# self.patient_info_xlsx = os.path.join(self.raw_input_folder, '2020_11_29_罕见病mdt_摘要_PID_27份病例.xlsx')
		# self.input_folder = input_folder or os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'pumc_pk_27')
		# self.pid_diag_codes_csv = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', '2021_05_19_pid_diag_codes_pumc_pk_27.xlsx')
		# self.TEST_PIDS_JSON = os.path.join(DATA_PATH, 'raw', 'PUMC_PK', 'test_pids_pumc_pk_27.json')
		# self.test_pids = None




		self.patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'PUMC_PK')
		os.makedirs(self.patient_save_folder, exist_ok=True)
		self.test_patient_save_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test')


	#def get_patient_info_list(self,pk_type=76):
	def get_patient_info_list(self):
		"""
		Returns:
			list: [{'PID': str, 'NAME': str, 'DIAG': str, 'CASE_ID': str, 'DEPARTMENT': str}]
		"""
		df = pd.read_excel(self.patient_info_xlsx, dtype=str)
		ret_list = []

		print('df===================',df)


		## PUMCH-ADM
		# for idx, row in df.iterrows():
		# 	ret_list.append({
		# 		'PID': row['PID'].strip(),
		# 		'NAME': row['姓名'].strip(),
		# 		'DIAG': row['诊断'].strip(),
		# 		'CASE_ID': row['病案号'].strip(),
		# 		'DEPARTMENT': row['科室'].strip()
		# 	})
		# # return ret_list
		# #



		for idx, row in df.iterrows():
			ret_list.append({
				'PID': row['PID'].strip(),
				'NAME': row['姓名'].strip(),
				'DIAG': row['确诊'].strip()
			})

		return ret_list



	def preprocess_raw(self,):
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
					print('name = {}; name in json = {}'.format(name, input_dict_list[0]['姓名']['RAW_TEXT']))	#
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

		pd.DataFrame(df_dict).to_excel(self.pid_diag_codes_csv, index=False,columns=['PID', 'NAME', 'DIAG_TEXT', 'DIS_CLASSES', 'DIS_CODES', 'DIS_CODES_CNS'])


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


	# todo 2020 11 29 第二个pumc pk 实验, mdt 27份
	base_folder = '/home/xhmao19/mxh19_personal/project/hy_works/2020_10_20_RareDisease-master/bert_syn_project/data/preprocess/'









	#
	score_thres_list = ['2021_04_28_wl_pumc_pk_json']


	#

	score_thres_list =['20220623_19cases']









	for i_paras in score_thres_list:

		outfolder_name = ''
		outfolder_name += i_paras







		paras = [(outfolder_name,os.path.join(base_folder, 'pumch_genotype_phenotype/'
														  + outfolder_name))]







		#
		# fields = [ '题目','专家导读','主诉', '现病史', '既往史','个人史', '月经婚育史', '家族史', '查体', '辅助检查',
		# 				  '初步诊断','手术治疗','总结病史','目前诊断','结局及转归','专家点评'
		# 				  ]


		fields = [ '主诉', '现病史', '既往史','个人史',
				  '查体', '辅助检查','初步诊断','总结病史','目前诊断','专科情况']



		hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])




		for mark, input_folder in paras:


			pg = PUMCPKPatientGenerator(fields=fields, mark=mark, input_folder=input_folder, hpo_reader=hpo_reader)


			#pg.show_pid_diag_csv()


			pg.process()
			print('....................Done the po.process  ................')


