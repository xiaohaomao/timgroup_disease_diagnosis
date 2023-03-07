

import pandas as pd
import random as rd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import xlrd
import re
import json
import os
from copy import deepcopy

from core.reader import HPOReader, HPOFilterDatasetReader, OMIMReader, OrphanetReader
from core.utils.utils import get_all_ancestors, get_all_descendents, slice_list_with_rm_set, transform_type, get_all_descendents_for_many
from core.utils.utils import get_all_ancestors_for_many, delete_redundacy, check_load_save, unique_list
from core.utils.constant import DATA_PATH, GENDER_FEMALE, GENDER_MALE, GENDER_NONE, JSON_FILE_FORMAT
from core.patient.patient_generator import PatientGenerator
from core.explainer import LabeledDatasetExplainer, Explainer

class SimPatientProbReader(object):
	def __init__(self):
		self.DB_NAME_MAP = {'MIM': 'OMIM', 'ORPHA': 'ORPHA'}
		self.COL_NAMES = ['NAME', 'CODE', 'PROB']
		self.PROB_MAP = {
			'very rare': 0.01, 'rare': 0.05, 'occasional': 0.075, 'main': 0.25,
			'frequent': 0.33, 'typical': 0.5, 'common': 0.5, 'hallmark': 0.9
		}
		self.sheet_name_pattern = re.compile('^Table S\d+\|(\w+) (\d+)$')
		self.code_cell_split_pattern = re.compile('(\\\\and/or|\\\\or/and|\\\\and|\\\\or)')
		self.HPO_PROB_XLSX_PATH = os.path.join(DATA_PATH, 'raw', 'simulation', 'Clinical_Diagnostics_in_Human_Genetics_with_Se_man.xlsx')
		self.DIS_TO_HPO_PROB_JSON = os.path.join(DATA_PATH, 'raw', 'simulation', 'dis_to_hpo_prob.json')
		self.dis_to_prob_item = None # {dis_code: [([hpo_code1, hpo_code2], prob), ([hpo_code1, hpo_code2], prob, GENDER_FEMALE),  (hpo_code, prob), (hpo_code, prob, GENDER_MALE)...]}
		self.IGNORE_ID = {'HP:0005833', 'UPDATE'}
		self.id_normalizer = {}  # {allId: mainId}
		self.DIS_NUM = 44


	def _get_id_normalizer(self):
		"""
		Returns:
			dict: {allId: mainId}
		"""
		id_map_main_id = {}
		hpo_reader = HPOReader()
		for mainId, info_dict in hpo_reader.get_hpo_dict().items():
			id_map_main_id[mainId] = mainId
			if 'ALT_ID' in info_dict:
				for alt_id in info_dict['ALT_ID']:
					id_map_main_id[alt_id] = mainId
		return id_map_main_id


	def id_normalize(self, hpo_code):
		if not self.id_normalizer:
			self.id_normalizer = self._get_id_normalizer()
		if hpo_code in self.IGNORE_ID:
			return None
		return self.id_normalizer[hpo_code]


	def _get_dis_code_from_sheet_name(self, sheet_name):
		match_obj = self.sheet_name_pattern.match(sheet_name)
		db_name = match_obj.group(1)  # MIM or ORPHA
		dis_id_num_str = match_obj.group(2)
		return self.DB_NAME_MAP[db_name] + ':' + dis_id_num_str


	def _prob_str_to_float(self, str):
		if type(str) is float:
			return str
		str = str.strip()
		if str in self.PROB_MAP:
			return self.PROB_MAP[str]
		match_obj = re.match('^(\d+)/(\d+)$', str)
		if match_obj:
			return int(match_obj.group(1)) / int(match_obj.group(2))
		match_obj = re.match('^(\d+.?\d+)%$', str)
		if match_obj:
			return float(match_obj.group(1)) / 100
		raise Exception('abnormal value of probability cell: ' + str)


	def _get_gender(self, name_cell_str):
		index_of_colon = name_cell_str.find(':')
		if index_of_colon == -1:
			return GENDER_NONE
		gender = name_cell_str[:index_of_colon].strip().upper()
		if gender == 'MALE':
			return GENDER_MALE
		if gender == 'FEMALE':
			return GENDER_FEMALE
		raise Exception('gender match wrong:' + name_cell_str + '->' + gender)


	def _read_sheet(self, sheet, id_normalizer):
		"""
		Args:
			sheet (xlrd.sheet.Sheet)
		Returns:
			list: [([hpo_code1, hpo_code2], prob), (hpo_code, prob), ...]
		"""
		def make_ret_item(hpo_code_or_list, prob_num, gender):
			if gender == GENDER_NONE:
				return (hpo_code_or_list, prob_num)
			return (hpo_code_or_list, prob_num, gender)
		prob_list = []
		for row in range(sheet.nrows):
			# print(sheet.name, row)
			item = {self.COL_NAMES[col]:sheet.cell(row, col).value for col in range(sheet.ncols)}
			gender = self._get_gender(item['NAME'])
			prob_num = self._prob_str_to_float(item['PROB'])
			temp_list = self.code_cell_split_pattern.split(item['CODE'])
			temp_list = [str.strip() for str in temp_list]
			if len(temp_list) > 1 and temp_list[1] == '\\and':
				hpo_list = slice_list_with_rm_set([self.id_normalize(temp_list[i]) for i in range(0, len(temp_list), 2)], remove_set={None})
				if hpo_list:
					prob_list.append(make_ret_item(hpo_list, prob_num, gender))
			else:
				for i in range(0, len(temp_list), 2):
					hpo_code = self.id_normalize(temp_list[i])
					if hpo_code:
						prob_list.append(make_ret_item(hpo_code, prob_num, gender))
		return prob_list


	@check_load_save('dis_to_prob_item', 'DIS_TO_HPO_PROB_JSON', JSON_FILE_FORMAT)
	def get_dis_to_prob_item(self):
		"""
		Returns:
			dict: {dis_code: ([hpo_code1, hpo_code2], prob), (hpo_code, prob), ...]}
		"""
		book = xlrd.open_workbook(self.HPO_PROB_XLSX_PATH)
		dis_to_prob_item = {}
		id_normalizer = self._get_id_normalizer()
		for sheet in book.sheets():
			dis_code = self._get_dis_code_from_sheet_name(sheet.name)
			dis_to_prob_item[dis_code] = self._read_sheet(sheet, id_normalizer)
		assert len(dis_to_prob_item) == self.DIS_NUM
		return dis_to_prob_item


class SimPatientGenerator(PatientGenerator):
	def __init__(self, hpo_reader=HPOReader()):
		super(SimPatientGenerator, self).__init__(hpo_reader)
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'SIMULATION')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.ORIGIN_PATIENTS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'origin.json')
		self.ORIGIN_STATS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'origin_statistics.json')
		self.NOISE_PATIENT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'noise.json')
		self.NOISE_STATS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'noise_statistics.json')
		self.IMPRECISION_PATIENT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'imprecision.json')
		self.IMPRECISION_STATS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'imprecision_statistics.json')
		self.IMPRECISION_NOISE_PATIENT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'impre_noise.json')
		self.IMPRECISION_NOISE_STATS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'impre_noise_statistics.json')
		self.NOISE_IMPRECISION_PATIENT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'noise_impre.json')
		self.NOISE_IMPRECISION_STATS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'noise_impre_statistics.json')

		self.DISEASE_MAP_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'simulation', 'disease_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(DATA_PATH, 'raw', 'simulation', 'disease_map.csv')

		self.dis_to_prob_item = SimPatientProbReader().get_dis_to_prob_item() # {dis_code: [([hpo_code1, hpo_code2], prob), ([hpo_code1, hpo_code2], prob, GENDER_FEMALE),  (hpo_code, prob), (hpo_code, prob, GENDER_MALE)...]}
		self.dis_to_all_labels = self.get_dis_code_map_all_labels()
		self.hpo_dict = HPOReader().get_hpo_dict()  # {CODE: {'NAME': .., 'IS_A': [], 'CHILD': [], ...}, ...}


	def get_noise_dataset_path(self, noise_prop, pa_num):
		return os.path.join(self.PREPROCESS_FOLDER, f'noise_{noise_prop}_{pa_num}.json')


	def get_noise_dataset_statistic_path(self, pa_path):
		prefix, postfix = os.path.splitext(pa_path)
		return prefix + '_statistics' + postfix


	def get_all_dis_codes(self):
		return sorted(unique_list([dis_code for dis_code in self.dis_to_prob_item]))


	def gen_disease_auto_match_csv(self):
		omim_reader = OMIMReader()
		omim_to_cns_info = {omim:info['CNS_NAME'] for omim, info in omim_reader.get_cns_omim().items()}
		omim_to_eng_info = {omim:info['ENG_NAME'] for omim, info in omim_reader.get_omim_dict().items()}
		orpha_reader = OrphanetReader()
		orpha_to_cns_info = {orpha:info['CNS_NAME'] for orpha, info in orpha_reader.get_cns_orpha_dict().items()}
		orpha_to_eng_info = {orpha:info['ENG_NAME'] for orpha, info in orpha_reader.get_orpha_dict().items()}
		omim_to_exact_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'E'})
		omim_to_broad_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'NTBT'})
		dis_codes = self.get_all_dis_codes()

		row_infos = []
		for omim_code in dis_codes:
			if not omim_code.startswith('OMIM'): continue
			orpha_codes = omim_to_exact_orpha.get(omim_code, [])
			assert len(orpha_codes) < 2
			if orpha_codes:
				orpha_code, link_type = orpha_codes[0], 'E'
			else:
				orpha_codes = omim_to_broad_orpha.get(omim_code, [])
				assert len(orpha_codes) < 2
				if orpha_codes:
					orpha_code, link_type = orpha_codes[0], 'B'
				else:
					orpha_code, link_type = '', ''
			row_dict = {
				'OMIM_CODE': omim_code,
				'OMIM_ENG': omim_to_eng_info.get(omim_code, ''),
				'OMIM_CNS': omim_to_cns_info.get(omim_code, ''),
				'ORPHA_LINK': link_type,
				'ORPHA_CODE': orpha_code,
				'ORPHA_ENG': orpha_to_eng_info.get(orpha_code, ''),
				'ORPHA_CNS': orpha_to_cns_info.get(orpha_code, ''),
			}
			row_infos.append(row_dict)
		row_infos.append({
			'OMIM_CODE': '',
			'OMIM_ENG': '',
			'OMIM_CNS': '',
			'ORPHA_LINK': '',
			'ORPHA_CODE': 'ORPHA:51',
			'ORPHA_ENG': orpha_to_eng_info.get('ORPHA:51', ''),
			'ORPHA_CNS':orpha_to_cns_info.get('ORPHA:51', ''),
		})

		pd.DataFrame(row_infos).to_csv(self.DISEASE_MAP_AUTO_CSV, index=False,
			columns=['OMIM_CODE', 'OMIM_ENG', 'OMIM_CNS', 'ORPHA_LINK', 'ORPHA_CODE', 'ORPHA_ENG', 'ORPHA_CNS'])


	def get_dis_code_map_all_labels(self, keep_links='all'):
		"""
		Returns:
			dict: {dis_code: [dis_code1, ...]}
		"""
		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'B'}
		row_list = pd.read_csv(self.DISEASE_MAP_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_item in row_list:
			dis_code = row_item['OMIM_CODE'] or row_item['ORPHA_CODE']
			orpha_code, orpha_link = row_item['ORPHA_CODE'], row_item['ORPHA_LINK']
			ccrd_code, ccrd_link = row_item['CCRD_CODE'], row_item['CCRD_LINK']
			dis_codes = [dis_code]
			if orpha_code and orpha_link in keep_links:
				dis_codes.append(orpha_code)
			if ccrd_code and ccrd_link in keep_links:
				dis_codes.append(ccrd_code)
			ret_dict[dis_code] = dis_codes
		assert 'ORPHA:51' in ret_dict
		ret_dict['ORPHA:51'].extend([
			'OMIM:114100', 'OMIM:225750', 'OMIM:610181', 'OMIM:610329', 'OMIM:610333', 'OMIM:612952', 'OMIM:615010'])
		return ret_dict


	def gen_gender(self):
		if rd.random() < 0.5:
			return GENDER_MALE
		return GENDER_FEMALE


	def gen_single_patient(self, dis_code):
		"""
		Args:
			dis_code (str): code of disease
		Returns:
			list: patient [[hpo_code1, hpo_code2, ...], [dis_code1, ...], ...]
		"""
		hpo_list = []
		gender = self.gen_gender()

		while len(hpo_list) == 0:
			for item in self.dis_to_prob_item[dis_code]: # (hpo_code, prob, maybe gender info)
				if len(item) == 3 and item[2] != gender:
					continue
				if rd.random() < item[1]:
					if type(item[0]) is list:
						hpo_list.extend(item[0])
					else:
						hpo_list.append(item[0])
		hpo_list = self.process_pa_hpo_list(hpo_list, reduce=True)
		dis_codes = self.dis_to_all_labels[dis_code]
		return [hpo_list, dis_codes]


	def generate_patients(self, patient_number=100):

		all_dis_codes = self.get_all_dis_codes()
		pa_dis_codes = [dis_code for dis_code in all_dis_codes for _ in range(patient_number)]
		patients = []
		with Pool() as pool:
			for patient in tqdm(pool.imap(self.gen_single_patient, pa_dis_codes, chunksize=200), total=len(pa_dis_codes), leave=False):
				patients.append(patient)
		return patients


	def generate_single_noise_patient(self, phe_list, noise_prop, dis_code):
		"""
		Args:
			phe_list (list): [hpo_code1, hpo_code2, ...]
		Returns:
			list: [hpo_code1, hpo_code2, ...]
		"""
		exclude_code_set = set()
		for item in self.dis_to_prob_item[dis_code]:
			hpo_codes = item[0] if isinstance(item[0], list) else [item[0]]
			exclude_code_set.update(get_all_ancestors_for_many(hpo_codes, self.hpo_dict))
			exclude_code_set.update(get_all_descendents_for_many(hpo_codes, self.hpo_dict))
		add_num = int(len(phe_list) * noise_prop)
		adding_hpo_list = rd.sample([code for code in self.hpo_dict.keys() if code not in exclude_code_set], add_num)
		hpo_list = self.process_pa_hpo_list(phe_list + adding_hpo_list, reduce=True)
		return hpo_list


	def multi_gsnp_wrap(self, paras):
		phe_list, noise_prop, dis_code = paras
		return self.generate_single_noise_patient(phe_list, noise_prop, dis_code)


	def generate_noise_patients(self, patients, noise_prop=0.5):
		"""
		Args:
			patients (list): [(phe_list, dis_code), ...], phe_list = [hpo_code1, ...]
		Returns:
			list: [(patient, dis_code), ...]
		"""
		def get_mark_dis_code(pa_dis_codes):
			for dis_code in pa_dis_codes:
				if dis_code in self.dis_to_prob_item:
					return dis_code
			assert False
		patients = deepcopy(patients)
		phe_lists = []
		paras = [(phe_list, noise_prop, get_mark_dis_code(pa_dis_codes)) for phe_list, pa_dis_codes in patients]
		with Pool() as pool:
			for phe_list in tqdm(pool.imap(self.multi_gsnp_wrap, paras, chunksize=200), total=len(paras), leave=False):
				phe_lists.append(phe_list)
		return [(phe_lists[i], patients[i][1]) for i in range(len(patients))]


	def generate_single_imprecision_patient(self, phe_list, replace_prob):
		"""
		Args:
			phe_list (list): [hpo_code1, hpo_code2, ...]
		Returns:
			list: [hpo_code1, hpo_code2, ...]
		"""
		new_phe_lists = []
		for hpo_code in phe_list:
			ancestor_set = slice_list_with_rm_set(
				get_all_ancestors(hpo_code, self.hpo_dict),
				{hpo_code, 'HP:0000001', 'HP:0000118'}   # self; All; Phenotypic abnormality
			)
			if len(ancestor_set) > 0 and rd.random() < replace_prob:
				new_phe_lists.append(rd.choice(ancestor_set))
			else:
				new_phe_lists.append(hpo_code)
		new_phe_lists = self.process_pa_hpo_list(new_phe_lists, reduce=True)
		return new_phe_lists


	def multi_gsip_wrap(self, paras):
		phe_list, replace_prob = paras
		return self.generate_single_imprecision_patient(phe_list, replace_prob)


	def generate_imprecision_patients(self, patients, replace_prob=1.0):
		"""
		Args:
			patients (list): [(phe_list, dis_code), ...], phe_list = [hpo_code1, ...]
		Returns:
			list: [(patient, dis_code), ...]
		"""
		patients = deepcopy(patients)
		phe_lists = []
		paras = [(phe_list, replace_prob) for phe_list, _ in patients]
		with Pool() as pool:
			for phe_list in tqdm(pool.imap(self.multi_gsip_wrap, paras, chunksize=200), total=len(paras), leave=False):
				phe_lists.append(phe_list)
		return [(phe_lists[i], patients[i][1]) for i in range(len(patients))]


	def delete_redundacy(self, phe_list):
		return self.process_pa_hpo_list(phe_list, reduce=True)
		# return delete_redundacy(phe_list, self.hpo_dict)


	def redundancy_patients(self, patients):
		"""
		Args:
			patients (list): [(phe_list, dis_code), ...], phe_list = [hpo_code1, ...]
		Returns:
			list: [(phe_list, dis_code), ...], phe_list = [hpo_code1, ...]
		"""
		phe_lists = []
		paras = [phe_list for phe_list, _ in patients]
		with Pool() as pool:
			for phe_list in tqdm(pool.imap(self.delete_redundacy, paras, chunksize=200), total=len(paras), leave=False):
				phe_lists.append(phe_list)
		return [(phe_lists[i], patients[i][1]) for i in range(len(patients))]


	def generate_all(self, patient_number=100, noise_prop=0.5, replace_prob=1.0):
		print('generating simulate patients...')
		patients = self.generate_patients(patient_number)
		json.dump(patients, open(self.ORIGIN_PATIENTS_JSON, 'w'), indent=2)
		explainer = LabeledDatasetExplainer(patients)
		json.dump(explainer.explain(),open(self.ORIGIN_STATS_JSON, 'w'), indent=2, ensure_ascii=False)
		print('done.')

		print('generating simulate patients with noise...')
		noise_pa = self.generate_noise_patients(patients, noise_prop)
		json.dump(noise_pa, open(self.NOISE_PATIENT_JSON, 'w'), indent=2)
		explainer = LabeledDatasetExplainer(noise_pa)
		json.dump(explainer.explain(), open(self.NOISE_STATS_JSON, 'w'), indent=2, ensure_ascii=False)
		print('done.')

		print('generating simulate patients with imprecision...')
		impre_pa = self.generate_imprecision_patients(patients, replace_prob)
		json.dump(impre_pa, open(self.IMPRECISION_PATIENT_JSON, 'w'), indent=2)
		explainer = LabeledDatasetExplainer(impre_pa)
		json.dump(explainer.explain(), open(self.IMPRECISION_STATS_JSON, 'w'), indent=2, ensure_ascii=False)
		print('done.')

		print('generating simulate patients with imprecision and noise...') #
		impre_noise_pa = self.generate_noise_patients(impre_pa, noise_prop)
		impre_noise_pa = self.redundancy_patients(impre_noise_pa)
		json.dump(impre_noise_pa, open(self.IMPRECISION_NOISE_PATIENT_JSON, 'w'), indent=2)
		explainer = LabeledDatasetExplainer(impre_noise_pa)
		json.dump(explainer.explain(), open(self.IMPRECISION_NOISE_STATS_JSON, 'w'), indent=2, ensure_ascii=False)
		print('done.')

		print('generating simulate patients with noise and imprecision...')
		noise_impre_pa = self.generate_imprecision_patients(noise_pa, replace_prob)
		noise_impre_pa = self.redundancy_patients(noise_impre_pa)
		json.dump(noise_impre_pa, open(self.NOISE_IMPRECISION_PATIENT_JSON, 'w'), indent=2)
		explainer = LabeledDatasetExplainer(noise_impre_pa)
		json.dump(explainer.explain(), open(self.NOISE_IMPRECISION_STATS_JSON, 'w'), indent=2, ensure_ascii=False)
		print('done.')


	def generate_different_noise_dataset(self, noise_probs):
		for noise_prob in noise_probs:
			print(f'generating simulate patients with noise {noise_prob}...')
			patients = json.load(open(self.ORIGIN_PATIENTS_JSON))
			noise_pa = self.generate_noise_patients(patients, noise_prob)
			save_path = self.get_noise_dataset_path(noise_prob, len(patients))
			json.dump(noise_pa, open(save_path, 'w'), indent=2)
			explainer = LabeledDatasetExplainer(noise_pa)
			json.dump(explainer.explain(), open(self.get_noise_dataset_statistic_path(save_path), 'w'), indent=2, ensure_ascii=False)
			print('done.')


	def generate_all_with_cns(self):
		json_paths = [
			self.ORIGIN_PATIENTS_JSON,
			self.NOISE_PATIENT_JSON,
			self.IMPRECISION_PATIENT_JSON,
			self.IMPRECISION_NOISE_PATIENT_JSON,
			self.NOISE_IMPRECISION_PATIENT_JSON
		]
		explainer = Explainer()
		for json_path in json_paths:
			save_json = os.path.splitext(json_path)[0] + '-cns.json'
			patients = json.load(open(json_path))
			json.dump(explainer.add_cns_info(patients), open(save_json, 'w'), indent=2, ensure_ascii=False)


def transform_to_train_data(json_path, txtPath, hpo_dict=None, extend=False):
	from core.reader.hpo_reader import HPOReader
	hpo_reader = HPOReader()
	hpo_map_rank = hpo_reader.get_hpo_map_rank()
	dis_map_rank = hpo_reader.get_dis_map_rank()
	data = json.load(open(json_path))
	if extend:
		for i in tqdm(range(len(data))):
			data[i][0] = list(get_all_ancestors_for_many(data[i][0], hpo_dict))
	data = transform_type(data, lambda x: str(hpo_map_rank[x]) if x in hpo_map_rank else str(dis_map_rank[x]))
	with open(txtPath, 'w') as fout:
		for i in range(len(data)):
			fout.write(data[i][1] + ' ' + ' '.join( data[i][0] ) + '\n')


if __name__ == '__main__':
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	pg = SimPatientGenerator(hpo_reader=hpo_reader)

	pg.generate_different_noise_dataset([20.0, 40.0])

