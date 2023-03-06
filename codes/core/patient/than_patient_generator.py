

import os
import pandas as pd
import re
import json
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from core.utils.utils import dict_list_add, dict_set_add, check_load_save
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.patient.patient_generator import PatientGenerator
from core.reader import HPOReader, HPOFilterDatasetReader


class ThanPatientGenerator(PatientGenerator):
	def __init__(self, hpo_reader=HPOReader()):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(ThanPatientGenerator, self).__init__(hpo_reader)
		self.RAW_FOLDER = os.path.join(DATA_PATH, 'raw', 'THAN')
		self.ALL_SUBSETS = ['Stavropoulos', 'Bone', 'Stelzer', 'Lee']
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, '')
		self.STR_MAP_CODES_AUTO_CSV = os.path.join(self.RAW_FOLDER, 'str_map_codes_auto.csv')
		self.STR_MAP_CODES_CSV = os.path.join(self.RAW_FOLDER, 'str_map_codes.csv')
		self.OMIM_MAP_ORPHA_CCRD_AUTO_CSV = os.path.join(self.RAW_FOLDER, 'omim_map_orpha_ccrd_auto.csv')
		self.OMIM_MAP_ORPHA_CCRD_CSV = os.path.join(self.RAW_FOLDER, 'omim_map_orpha_ccrd.csv')

		self.OUTPUT_PATIENT_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'THAN')
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)
		self.PATIENTS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'patients.json')
		self.PIDS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'pids.json')
		self.patients = None


	def get_raw_xlsx(self, subset_name):
		"""
		Args:
			subset_name: 'Bone' | 'Lee' | 'Stavropoulos' | 'Stelzer'
		"""
		return os.path.join(self.RAW_FOLDER, 'subset',subset_name+'.xlsx' )


	def read_xlsx(self, path):

		def get_hpo_list(cell_str):
			return re.findall('HP:\d+', cell_str)
		def get_omim_list(cell_str):
			mim_list = re.findall('MIM:\s*\d+', cell_str)
			return ['OMIM:' + mim.split(':').pop().strip() for mim in mim_list]
		def get_orpha_list(cell_str):
			obj_list = re.finditer('(OR:|ORPHA)\s*(\d+)', cell_str)
			return ['ORPHA:' + obj.group(2) for obj in obj_list] if obj_list else []
		def process_str(cell_str):
			return cell_str.replace('\n', ' ').strip()

		key_names = ['PID', 'HPO', 'DIAG_OMIM_STR', 'DIAG_ORPHA_STR', 'GENE', 'GENE_ID', 'LOCUS', 'INH']
		key2idx = {k: i for i, k in enumerate(key_names)}
		df = pd.read_excel(path).fillna('')
		ret_list = []
		for idx, row_se in df.iterrows():
			row_info = {}
			row_info['PID'] = row_se.iloc[key2idx['PID']].strip()
			row_info['HPO_LIST'] = get_hpo_list(row_se.iloc[key2idx['HPO']])
			row_info['OMIM_STR'] = process_str(row_se.iloc[key2idx['DIAG_OMIM_STR']])
			row_info['OMIM_LIST'] = get_omim_list(row_info['OMIM_STR'])
			row_info['ORPHA_STR'] = process_str(row_se.iloc[key2idx['DIAG_ORPHA_STR']])
			row_info['ORPHA_LIST'] = get_orpha_list(row_info['ORPHA_STR'])
			row_info['GENE'] = row_se.iloc[key2idx['GENE']]
			row_info['INH'] = row_se.iloc[key2idx['INH']]
			ret_list.append(row_info)
		return ret_list


	@check_load_save('patients', 'PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_patients(self):
		pid2patients = self._gen_patients(keep_links='all')
		pids, patients = zip(*pid2patients.items())
		json.dump(pids, open(self.PIDS_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients


	def _gen_patients(self, keep_links='all'):
		"""
		Args:
			keep_links (str or set)
		Returns:
			dict: {pid: patient, ...}; patient = [[hpo_code1, hpo_code2, ...], [dis_code1, dis_code2, ...]]
		"""
		row_infos = []
		for subset_name in self.ALL_SUBSETS:
			row_infos.extend(self.read_xlsx(self.get_raw_xlsx(subset_name)))
		str_map_code_link = self.get_str_to_code_link(keep_links=keep_links)
		omim_map_code_link = self.get_omim_map_code_link(keep_links=keep_links)
		pid2patients = OrderedDict()
		for row_info in row_infos:
			pid, hpo_list = row_info['PID'], row_info['HPO_LIST']
			omim_str, omim_list = row_info['OMIM_STR'], row_info.get('OMIM_LIST', [])
			if omim_list:
				diag_codes = deepcopy(omim_list)
				diag_codes.extend([map_code for omim in omim_list for map_code, link_type in omim_map_code_link.get(omim, [])])
			else:
				diag_codes = [map_code for map_code, link_type in str_map_code_link.get(omim_str, [])]
			diag_codes = np.unique(diag_codes).tolist()
			diag_codes = self.process_pa_dis_list(diag_codes)
			if not diag_codes:
				print('No diag codes: {}; {}'.format(pid, omim_str))
			hpo_list = self.process_pa_hpo_list(hpo_list)
			pid2patients[pid] = [hpo_list, diag_codes]
		return pid2patients


	def get_labels_set_with_all_eq_sources(self, sources):
		"""
		Returns:
			set: {sorted_dis_codes_tuple, ...}; sorted_dis_codes_tuple = (dis_code1, dis_code2, ...)
		"""
		patients = self._gen_patients(keep_links={'E'})
		return set([tuple(sorted(dis_codes)) for _, dis_codes in patients if self.diseases_from_all_sources(dis_codes, sources)])


	def get_keep_to_general_dis(self):
		"""
		Returns:
			dict: {keep_dis_code: [general_dis1, ...]}
		"""
		def get_code_with_mark(dis_codes, mark='OMIM:'):
			for dis_code in dis_codes:
				if dis_code.startswith('OMIM:'):
					return dis_code
			return None
		str_map_code_link = self.get_str_to_code_link(keep_links='all')
		omim_map_code_link = self.get_omim_map_code_link(keep_links={'B'})
		ret_dict = {}
		for str, code_link_list in str_map_code_link.items():
			e_codes = [code for code, link_type in code_link_list if link_type == 'E']
			center_code = get_code_with_mark(e_codes, 'OMIM:') or get_code_with_mark(e_codes, 'ORPHA:')
			b_codes = [code for code, link_type in code_link_list if link_type == 'B']
			if len(b_codes) > 0 and center_code is not None:
				for b_code in b_codes:
					dict_set_add(center_code, b_code, ret_dict)
		for center_code, code_link_list in omim_map_code_link.items():
			b_codes = [code for code, link_type in code_link_list if link_type == 'B']
			if len(b_codes) > 0 and center_code is not None:
				for b_code in b_codes:
					dict_set_add(center_code, b_code, ret_dict)
		return {keep_code: list(rm_code_set) for keep_code, rm_code_set in ret_dict.items()}


	def get_str_to_code_link(self, keep_links='all'):

		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'B', 'N'}
		row_list = pd.read_csv(self.STR_MAP_CODES_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_info in row_list:
			diag_str = row_info['STR'].strip()
			map_code = row_info['MAP_CODE'].strip()
			link_type = row_info['LINK_TYPE'].strip()
			if link_type in keep_links:
				dict_list_add(diag_str, (map_code, link_type), ret_dict)
		return ret_dict


	def gen_str_to_codes_csv(self):
		row_infos = []
		for subset_name in self.ALL_SUBSETS:
			row_infos.extend(self.read_xlsx(self.get_raw_xlsx(subset_name)))
		write_row_infos = []
		for row_info in row_infos:
			if row_info['OMIM_LIST']:
				continue
			write_row_infos.append({
				'STR': row_info['OMIM_STR'],
				'LINK_TYPE': '',
				'MAP_CODE': '',
				'MAP_ENG': '',
				'MAP_CNS':'',
			})
		pd.DataFrame(write_row_infos).to_csv(self.STR_MAP_CODES_AUTO_CSV, index=False,
			columns=['STR', 'LINK_TYPE', 'MAP_CODE', 'MAP_ENG', 'MAP_CNS'])


	def get_omim_map_code_link(self, keep_links='all'):

		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'B', 'N'}
		row_list = pd.read_csv(self.OMIM_MAP_ORPHA_CCRD_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_info in row_list:
			omim_code = row_info['OMIM_CODE'].strip()
			map_code = row_info['MAP_CODE'].strip()
			link_type = row_info['LINK_TYPE'].strip()
			if link_type in keep_links:
				dict_list_add(omim_code, (map_code, link_type), ret_dict)
		return ret_dict


	def gen_disease_auto_match_csv(self):
		from core.reader import OrphanetReader, OMIMReader, RDReader, HPOReader, CCRDReader
		def get_equal_code(from_code, to_prefix):
			if from_code not in source2rd:
				print('Not found:', from_code)
				return None
			rd = source2rd[from_code]
			for source_code in rd2source[rd]:
				if source_code.startswith(to_prefix):
					return source_code
			return None
		def check_omim_list(omim_codes):
			hpo_reader = HPOReader()
			dis_set = set(hpo_reader.get_dis_list())
			for omim_code in omim_codes:
				if omim_code not in dis_set:
					print('Not found:', omim_code)

		rd_reader = RDReader()
		source2rd, rd2source = rd_reader.get_source_to_rd(), rd_reader.get_rd_to_sources()

		row_infos = []
		for subset_name in self.ALL_SUBSETS:
			row_infos.extend(self.read_xlsx(self.get_raw_xlsx(subset_name)))
		omim_codes = [omim_code for row_info in row_infos for omim_code in row_info['OMIM_LIST']]
		check_omim_list(omim_codes)
		dis2name_hpo_provide = HPOReader().get_dis_to_name()

		omim_reader = OMIMReader()
		omim_to_cns_info = {omim:info['CNS_NAME'] for omim, info in omim_reader.get_cns_omim().items()}
		omim_to_eng_info = {code: eng_name for code, eng_name in dis2name_hpo_provide.items() if code.startswith('OMIM:')}
		omim_to_eng_info.update({omim:info['ENG_NAME'] for omim, info in omim_reader.get_omim_dict().items()})

		orpha_reader = OrphanetReader()
		orpha_to_cns_info = {orpha:info['CNS_NAME'] for orpha, info in orpha_reader.get_cns_orpha_dict().items()}
		orpha_to_eng_info = {code: eng_name for code, eng_name in dis2name_hpo_provide.items() if code.startswith('ORPHA:')}
		orpha_to_eng_info.update({orpha:info['ENG_NAME'] for orpha, info in orpha_reader.get_orpha_dict().items()})
		omim_to_exact_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'E'})
		omim_to_broad_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'NTBT'})
		omim_to_narrow_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'BTNT'})

		ccrd_dict = CCRDReader().get_ccrd_dict()
		ccrd_to_cns_info = {ccrd: info['CNS_NAME'] for ccrd, info in ccrd_dict.items()}
		ccrd_to_eng_info = {ccrd: info['ENG_NAME'] for ccrd, info in ccrd_dict.items()}

		omim_e_ccrd = {}
		for omim in omim_codes:
			ccrd = get_equal_code(omim, 'CCRD:')
			if ccrd:
				omim_e_ccrd[omim] = ccrd

		code_link_list = []
		for omim_code in omim_codes:
			tmp_code_link = []
			tmp_code_link.extend([(omim_code, orpha_code, 'E') for orpha_code in omim_to_exact_orpha.get(omim_code, [])])
			tmp_code_link.extend([(omim_code, orpha_code, 'B') for orpha_code in omim_to_broad_orpha.get(omim_code, [])])
			tmp_code_link.extend([(omim_code, orpha_code, 'N') for orpha_code in omim_to_narrow_orpha.get(omim_code, [])])
			tmp_code_link = tmp_code_link or [(omim_code, '', '')]
			code_link_list.extend(tmp_code_link)
			tmp_code_link = []
			if omim_code in omim_e_ccrd:
				tmp_code_link.append((omim_code, omim_e_ccrd[omim_code], 'E'))
			tmp_code_link = tmp_code_link or [(omim_code, '', '')]
			code_link_list.extend(tmp_code_link)

		row_infos = []
		last_omim_code = ''
		for omim_code, map_code, link_type in code_link_list:
			if omim_code != last_omim_code:
				row_infos.append({})
				last_omim_code = omim_code
			row_infos.append({
				'OMIM_CODE':omim_code,
				'OMIM_ENG':omim_to_eng_info.get(omim_code, ''),
				'OMIM_CNS':omim_to_cns_info.get(omim_code, ''),
				'LINK_TYPE':link_type,
				'MAP_CODE':map_code,
				'MAP_ENG':orpha_to_eng_info.get(map_code, '') if map_code.startswith('ORPHA:') else ccrd_to_eng_info.get(map_code, ''),
				'MAP_CNS':orpha_to_cns_info.get(map_code, '') if map_code.startswith('ORPHA:') else ccrd_to_cns_info.get(map_code, ''),
			})
		pd.DataFrame(row_infos).to_csv(self.OMIM_MAP_ORPHA_CCRD_AUTO_CSV, index=False,
			columns=['OMIM_CODE', 'OMIM_ENG', 'OMIM_CNS', 'LINK_TYPE', 'MAP_CODE', 'MAP_ENG', 'MAP_CNS'])



if __name__ == '__main__':
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	pg = ThanPatientGenerator(hpo_reader)



