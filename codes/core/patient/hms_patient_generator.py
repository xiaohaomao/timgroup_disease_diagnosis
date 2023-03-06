

import os
import pandas as pd
import re
import json
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from core.utils.utils import dict_list_add, dict_set_add, check_load_save, zip_sorted
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.patient.patient_generator import PatientGenerator
from core.reader import HPOReader, HPOFilterDatasetReader


class HmsPatientGenerator(PatientGenerator):
	def __init__(self, hpo_reader=HPOReader()):
		super(HmsPatientGenerator, self).__init__(hpo_reader=hpo_reader)
		self.RAW_FOLDER = os.path.join(DATA_PATH, 'raw', 'HMS')
		self.RAW_CSV = os.path.join(self.RAW_FOLDER, 'hms_patients.csv')
		self.RAW_JSON = os.path.join(self.RAW_FOLDER, 'hms_patients.json')
		self.SYMPTOM_MAP_HPO_AUTO_CSV = os.path.join(self.RAW_FOLDER, 'symptom_map_hpo_auto.csv')
		self.SYMPTOM_MAP_HPO_CSV = os.path.join(self.RAW_FOLDER, 'symptom_map_hpo.csv')
		self.DISEASE_MAP_AUTO_CSV = os.path.join(self.RAW_FOLDER, 'disease_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(self.RAW_FOLDER, 'disease_map.csv')

		self.OUTPUT_PATIENT_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'HMS')
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)
		self.PATIENTS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'patients.json')
		self.PIDS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'pids.json')
		self.patients = None

		self.col_names = ['VISIT', 'SEX', 'AGE', 'DIAG', 'PRF', 'SYMPTOMS', 'ARF', 'N_SYMPTOMS']
		self.col2idx = {col_name: i for i, col_name in enumerate(self.col_names)}
		self.pid_to_multi_visit = None


	@check_load_save('patients', 'PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_patients(self):
		pid2patients = self._gen_patients(keep_dis_links='all', keep_hpo_links='all')
		pids, patients = zip(*pid2patients.items())
		json.dump(pids, open(self.PIDS_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients


	def _gen_patients(self, keep_dis_links='all', keep_hpo_links='all'):
		"""
		Args:
			keep_dis_links (str or set)
			keep_hpo_links (str or set)
		Returns:
			dict: {pid: patient, ...}; patient = [[hpo_code1, hpo_code2, ...], [dis_code1, dis_code2, ...]]
		"""
		pid_to_multi_visit = self.read_csv()
		pid_to_diag_codes = self.get_pid_to_diag_codes(pid_to_multi_visit, keep_dis_links)
		pid_to_hpo_codes = self.get_pid_to_hpo_codes(pid_to_multi_visit, keep_hpo_links)
		pid2patients = OrderedDict()
		for pid in pid_to_multi_visit:
			diag_codes = self.process_pa_dis_list(pid_to_diag_codes[pid])
			hpo_list = self.process_pa_hpo_list(pid_to_hpo_codes[pid])
			if not diag_codes or not hpo_list:
				print('No diag codes or hpo codes: {}; {}; {} '.format(pid, diag_codes, hpo_list))
			pid2patients[pid] = [hpo_list, diag_codes]
		return pid2patients


	def get_pid_to_hpo_codes(self, pid_to_multi_visit=None, keep_links='all'):
		"""
		Returns:
			dict: {PID: [hpo_code1, hpo_code2, ...]}
		"""
		pid_to_multi_visit = pid_to_multi_visit or self.read_csv()
		str_to_hpocode_link = self.get_str_to_hpocode_link(keep_links=keep_links)
		ret_dict = {}
		for pid, multi_visit_dict in pid_to_multi_visit.items():
			symptoms = []
			for visit_dict in multi_visit_dict.values():
				symptoms.extend(visit_dict['SYMPTOMS'])
			hpo_list = []
			for sym_str in symptoms:
				hpo_list.extend([hpo for hpo, link in str_to_hpocode_link.get(sym_str, [])])
			hpo_list = np.unique(hpo_list).tolist()
			ret_dict[pid] = hpo_list
		return ret_dict


	def get_pid_to_diag_codes(self, pid_to_multi_visit=None, keep_links='all'):
		"""
		Returns:
			dict: {PID: [dis_code1, dis_code2, ...]}
		"""
		pid_to_multi_visit = pid_to_multi_visit or self.read_csv()
		str_to_dis_code_link = self.get_str_to_discode_link(keep_links)
		ret_dict = {}
		for pid, multi_visit_dict in pid_to_multi_visit.items():
			diag_strs = multi_visit_dict['DIAG_VISIT']['DIAG']
			if pid == '50':
				diag_codes = ['ORPHA:49041']
			else:
				diag_codes = []
				for diag_str in diag_strs:
					if diag_str not in str_to_dis_code_link:
						print(diag_str)
					diag_codes.extend([dis_code for dis_code, link_type in str_to_dis_code_link.get(diag_str, [])])
				diag_codes = np.unique(diag_codes).tolist()
			ret_dict[pid] = diag_codes
		return ret_dict


	def get_str_to_discode_link(self, keep_links='all'):
		"""
		Args:
			keep_links (str or set)
		Returns:
			dict: {diag_str: [(dis_code1, link_type), ...]}
		"""
		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'B', 'N'}
		row_list = pd.read_csv(self.DISEASE_MAP_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_info in row_list:
			diag_str = row_info['STR'].strip()
			map_code = row_info['MAP_CODE'].strip()
			link_type = row_info['LINK_TYPE'].strip()
			if link_type in keep_links:
				dict_list_add(diag_str, (map_code, link_type), ret_dict)
		return ret_dict


	def gen_str_to_discodes_csv(self):
		pid_to_multi_visit = self.read_csv()
		all_diag_str = sorted(np.unique([multi_visit_dict['DIAG_VISIT']['DIAG'] for pid, multi_visit_dict in pid_to_multi_visit.items()]).tolist())
		write_row_infos = []
		for diag in all_diag_str:
			write_row_infos.append({
				'STR': diag,
				'LINK_TYPE':'',
				'MAP_CODE':'',
				'MAP_ENG':'',
				'MAP_CNS':'',
			})
		pd.DataFrame(write_row_infos).to_csv(self.DISEASE_MAP_AUTO_CSV, index=False,
			columns=['STR', 'LINK_TYPE', 'MAP_CODE', 'MAP_ENG', 'MAP_CNS'])


	def get_str_to_hpocode_link(self, keep_links='all'):
		"""
		Args:
			keep_links (str or set)
		Returns:
			dict: {symp_str: [(hpo_code1, link_type), ...]}
		"""
		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'U', 'B', 'N'}
		row_list = pd.read_csv(self.SYMPTOM_MAP_HPO_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_info in row_list:
			symp_str = row_info['SYMPTOM'].strip()
			map_code = row_info['HPO'].strip()
			link_type = row_info['TYPE'].strip() or 'E'
			if map_code and link_type in keep_links:
				dict_list_add(symp_str, (map_code, link_type ), ret_dict)
		return ret_dict


	def gen_symptom_auto_match_csv(self):
		hpo_dict = self.hpo_reader.get_hpo_dict()
		chpo_dict = self.hpo_reader.get_chpo_dict()
		syn_to_hpo = {syn.lower(): hpo for syn, hpo in self.hpo_reader.get_syn2hpo().items()}

		pid_to_multi_visit = self.read_csv()
		all_symptoms = []
		for pid, multi_visit_dict in pid_to_multi_visit.items():
			for visit_dict in multi_visit_dict.values():
				all_symptoms.extend(visit_dict['SYMPTOMS'])
		all_symptoms = sorted(np.unique(all_symptoms).tolist())

		hpos = [syn_to_hpo.get(symptom.lower(), '') for symptom in all_symptoms]
		eng_names = [hpo_dict.get(hpo, {}).get('ENG_NAME', '') for hpo in hpos]
		cns_names = [chpo_dict.get(hpo, {}).get('CNS_NAME', '') for hpo in hpos]
		hpos, symptoms, eng_names, cns_names = zip_sorted(hpos, all_symptoms, eng_names, cns_names)
		pd.DataFrame(
			{'SYMPTOM': symptoms, 'HPO': hpos, 'ENG_NAME': eng_names, 'CNS_NAME': cns_names},
			columns=['SYMPTOM', 'HPO', 'ENG_NAME', 'CNS_NAME'],
		).to_csv(self.SYMPTOM_MAP_HPO_AUTO_CSV, index=False)


	@check_load_save('pid_to_multi_visit', 'RAW_JSON', JSON_FILE_FORMAT)
	def read_csv(self):
		"""
		Returns:
			dict: {
				PID: {
					'T5F_VISIT': info_dict,
					'TF_VISIT': info_dict,
					'T5F_TF_VISIT': info_dict
					'DIAG_VISIT': info_dict
				}
			}, info_dict = {
				'SEX': int,
				'AGE': int,
				'DIAG': str,
				'PRF': str, # Present Risk Factors
				'SYMPTOMS': [str, ...],
				'ARF': str, # Absent Risk Factors
				'N_SYMPTOMS': [str, ...]
			}

		"""
		str_mat = pd.read_csv(self.RAW_CSV, header=None).fillna('').values.tolist()
		str_mat.append(['94', '', '', '', '', '', '', ''])
		pid_to_patient_mat = {} # {pid: patient_mat}

		last_idx, last_pid = 0, '1'
		for i in range(1, len(str_mat)):
			row = str_mat[i]
			match_obj = re.match('(\d+)', row[0].strip())
			if match_obj:
				pid_to_patient_mat[last_pid] = str_mat[last_idx: i]
				last_idx, last_pid = i, match_obj.group(1)
		ret_dict = {}
		for pid, patient_mat in pid_to_patient_mat.items():
			ret_dict[pid] = self.patient_mat_to_dict(patient_mat)
		return ret_dict


	def patient_mat_to_dict(self, patient_mat):
		"""
		Args:
			patient_mat (list)
		Returns:
			dict: {
				'T5F_VISIT': info_dict,
				'TF_VISIT': info_dict,
				'T5F_TF_VISIT': info_dict
				'DIAG_VISIT': info_dict
			}
		"""
		def get_visit_key(s):
			T5F = re.search('Top Five Fit', s) is not None
			TF = re.search('Top Fit', s) is not None
			DIAG = re.search('Diagnosis', s) is not None
			if DIAG:
				return 'DIAG_VISIT'
			if T5F and TF:
				return 'T5F_TF_VISIT'
			if T5F:
				return 'T5F_VISIT'
			if TF:
				return 'TF_VISIT'
			return None

		visit_key_idx = [] # [(key, idx), ...]
		for i, row in enumerate(patient_mat):
			k = get_visit_key(row[0])
			if k:
				visit_key_idx.append((k, i))
		ret_dict = {}
		visit_key_idx.append(('None', len(patient_mat)))
		for i in range(len(visit_key_idx) - 1):
			k, idx_b = visit_key_idx[i]
			_, idx_e = visit_key_idx[i+1]
			ret_dict[k] = self.visit_mat_to_dict(patient_mat[idx_b: idx_e])
		return ret_dict


	def visit_mat_to_dict(self, visit_mat):
		"""
		Args:
			visit_mat (list)
		Returns:
			dict: {
				'SEX': int,
				'AGE': int,
				'DIAG': str,
				'PRF': str, # Present Risk Factors
				'SYMPTOMS': [str, ...],
				'ARF': str, # Absent Risk Factors
				'N_SYMPTOMS': [str, ...]
			}
		"""
		return {
			'SEX': visit_mat[0][self.col2idx['SEX']].strip(),
			'AGE': visit_mat[0][self.col2idx['AGE']].strip(),
			'DIAG': [row[self.col2idx['DIAG']].strip() for row in visit_mat if row[self.col2idx['DIAG']].strip()],
			'PRF': visit_mat[0][self.col2idx['PRF']].strip(),
			'ARF': visit_mat[0][self.col2idx['ARF']].strip(),
			'SYMPTOMS': [row[self.col2idx['SYMPTOMS']].strip() for row in visit_mat if row[self.col2idx['SYMPTOMS']].strip()],
			'N_SYMPTOMS': [row[self.col2idx['N_SYMPTOMS']].strip() for row in visit_mat if row[self.col2idx['N_SYMPTOMS']].strip()]
		}


	def get_labels_set_with_all_eq_sources(self, sources):
		"""
		Returns:
			set: {sorted_dis_codes_tuple, ...}; sorted_dis_codes_tuple = (dis_code1, dis_code2, ...)
		"""
		patients = self._gen_patients()
		return set([tuple(sorted(dis_codes)) for _, dis_codes in patients if self.diseases_from_all_sources(dis_codes, sources)])


	def get_keep_to_general_dis(self):
		"""
		Returns:
			dict: {keep_dis_code: [general_dis1, ...]}
		"""
		return {}


if __name__ == '__main__':
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	pg = HmsPatientGenerator(hpo_reader=hpo_reader)

	pg.get_patients()