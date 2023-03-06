
import json
import os
import pandas as pd
import re

from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, unique_list, zip_sorted, dict_list_add
from core.draw.simpledraw import simple_dist_plot
from core.patient.patient_generator import PatientGenerator
from core.reader import OMIMReader, OrphanetReader, HPOReader


class RamedisPatientGenerator(PatientGenerator):
	def __init__(self, hpo_reader=HPOReader()):
		super(RamedisPatientGenerator, self).__init__(hpo_reader)
		self.SYMPTOM_MAP_HPO_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'symptom_map_hpo_auto.csv')
		self.SYMPTOM_MAP_HPO_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'symptom_map_hpo.csv')
		self.DISEASE_MAP_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'disease_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'disease_map.csv')
		self.sym_name_range_map_hpos = None
		self.remove_sym_types = set()  # note: need to delete self.RAMEDIS_PATIENTS_JSON if used
		self.LAB_FINDING_MAP_HPO_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'labfinding_map_hpo_auto.csv')
		self.LAB_FINDING_MAP_HPO_CSV = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'labfinding_map_hpo.csv')
		self.lab_finding_map_hpos = None
		self.remove_lab_types = set()
		self.INPUT_PATIENT_JSON = os.path.join(DATA_PATH, 'raw', 'RAMEDIS', 'ramedis_patients.json')
		self.pid_to_pitem = None

		self.OUTPUT_PATIENT_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'RAMEDIS')
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)
		self.RAMEDIS_PATIENTS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'patients.json')
		self.RAMEDIS_PIDS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'pids.json')
		self.patients = None
		self.RAMEDIS_DIAG_CONFIRM_PATIENTS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'diag_patients.json')
		self.RAMEDIS_DIAG_CONFIRM_PIDS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'diag_pids.json')
		self.diag_patients = None
		self.RAMEDIS_DIAG_CONFIRM_PATIENTS_NOT_BORN_SCREEN_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'diag_notscreen_patients.json')
		self.RAMEDIS_DIAG_CONFIRM_PIDS_NOT_BORN_SCREEN_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'diag_notscreen_pids.json')
		self.diag_not_screen_patients = None

		self.hpo_to_death_year_range = {
			'HP:0003811': (0.0, 28/365),   #
			'HP:0001522': (0.0, 2.0),   #
			'HP:0003819': (2.0, 10.0),  #
			'HP:0011421': (10.0, 19.0),   #
			'HP:0100613': (16.0, 40.0),   #
		}


	@check_load_save('patients', 'RAMEDIS_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_patients(self):
		patients, pids = self._get_patients([])
		json.dump(pids, open(self.RAMEDIS_PIDS_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients


	@check_load_save('diag_patients', 'RAMEDIS_DIAG_CONFIRM_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_diag_patients(self):
		patients, pids = self._get_patients([self.keep_pid_if_diag_confirm])
		json.dump(pids, open(self.RAMEDIS_DIAG_CONFIRM_PIDS_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients


	@check_load_save('diag_not_screen_patients', 'RAMEDIS_DIAG_CONFIRM_PATIENTS_NOT_BORN_SCREEN_JSON', JSON_FILE_FORMAT)
	def get_diag_not_screen_patients(self):
		patients, pids = self._get_patients([self.keep_pid_if_diag_confirm, self.keep_pid_if_not_born_screen])
		json.dump(pids, open(self.RAMEDIS_DIAG_CONFIRM_PIDS_NOT_BORN_SCREEN_JSON, 'w'), indent=2, ensure_ascii=False)
		return patients


	def _get_patients(self, keep_pid_filters):
		"""
		Args:
			keep_pid_filters (list): [func1, func2, ...]; keep pid if all func(pid) return True
		"""
		patients, pids = [], []
		pid_to_pitem = self.get_pid_to_pitem()
		omim_to_discodes = self.get_omim_map_all_labels()
		filter_num = len(keep_pid_filters)
		sym_name_range_map_hpos = self.get_sym_name_range_map_hpos()
		lab_findings_map_hpos = self.get_lab_finding_map_hpos()
		for pid in sorted(pid_to_pitem.keys(), key=lambda pid: int(pid)):
			p_item = pid_to_pitem[pid]
			if sum([f(pid) for f in keep_pid_filters]) != filter_num: continue
			dis_code = self.get_dis_code(p_item['MainData']['Diagnosis'])
			dis_codes = omim_to_discodes.get(dis_code, [])
			hpo_list = []
			for symItem in p_item['Symptoms']:
				hpo_list.extend(self.sym_item_to_hpos(symItem, sym_name_range_map_hpos))
			for labItem in p_item['Lab findings']:
				hpo_list.extend(self.lab_item_to_hpos(labItem, lab_findings_map_hpos))
			hpo_list = self.process_pa_hpo_list(hpo_list, reduce=False)
			patients.append([hpo_list, dis_codes])
			pids.append(pid)
		return patients, pids


	def keep_pid_if_diag_confirm(self, pid):
		return self.pid_to_pitem[pid]['MainData']['Diagnosis confirmed'] == 'y'


	def keep_pid_if_not_born_screen(self, pid):
		return self.pid_to_pitem[pid]['MainData']['Found in newborn screening'] != 'y'


	def get_pid_to_pitem(self):
		if self.pid_to_pitem is None:
			self.pid_to_pitem = json.load(open(self.INPUT_PATIENT_JSON))
		return self.pid_to_pitem


	def sym_item_to_hpos(self, symItem, sym_name_range_map_hpos):
		"""
		Args:
			symItem (dict): e.g. {"Name": "early death", "Age": "4.93 Month(s)", "Range": "yes"},
		Returns:
			list: [hpo_code1, hpo_code2, ...]
		"""
		ret_hpo_list = []
		sym_name, symAge, sym_range = symItem['Name'], symItem['Age'], symItem['Range']
		if not self.have_symptom(sym_range):
			return ret_hpo_list
		if sym_name == 'death' or 'early death' or 'sudden death' or 'previous deaths':
			ret_hpo_list.extend(self.death_age_to_hpos(symAge))
		ret_hpo_list.extend(sym_name_range_map_hpos.get((sym_name, sym_range), []))
		ret_hpo_list.extend(sym_name_range_map_hpos.get((sym_name, ''), []))
		return ret_hpo_list


	def get_sym_name_range_map_hpos(self):
		"""
		Returns:
			dict: {(sym_name, sym_range): [hpo_code, ...]}
		"""
		if self.sym_name_range_map_hpos:
			return self.sym_name_range_map_hpos
		ret_dict = {}
		row_list = pd.read_csv(self.SYMPTOM_MAP_HPO_CSV).fillna('').to_dict(orient='records')
		for row_item in row_list:
			if len(row_item['HPO'].strip()) == 0:
				continue
			if row_item['TYPE'].strip() in self.remove_sym_types:
				continue
			dict_list_add((row_item['SYMPTOM'].strip(), row_item['RANGE'].strip()), row_item['HPO'].strip(), ret_dict)
		self.sym_name_range_map_hpos = ret_dict
		return self.sym_name_range_map_hpos


	def gen_symptom_auto_match_csv(self):
		hpo_dict = self.hpo_reader.get_hpo_dict()
		chpo_dict = self.hpo_reader.get_chpo_dict()
		syn_to_hpo = {syn.lower(): hpo for syn, hpo in self.hpo_reader.get_syn2hpo().items()}

		pid_to_pitem = self.get_pid_to_pitem()
		symptoms = unique_list([symItem['Name'] for pid, p_item in pid_to_pitem.items() for symItem in p_item['Symptoms']])
		hpos = [syn_to_hpo.get(symptom.lower(), '') for symptom in symptoms]
		eng_names = [hpo_dict.get(hpo, {}).get('ENG_NAME', '') for hpo in hpos]
		cns_names = [chpo_dict.get(hpo, {}).get('CNS_NAME', '') for hpo in hpos]
		hpos, symptoms, eng_names, cns_names = zip_sorted(hpos, symptoms, eng_names, cns_names)

		pd.DataFrame(
			{'SYMPTOM': symptoms, 'HPO': hpos, 'ENG_NAME': eng_names, 'CNS_NAME': cns_names},
			columns=['SYMPTOM', 'HPO', 'ENG_NAME', 'CNS_NAME'],
		).to_csv(self.SYMPTOM_MAP_HPO_AUTO_CSV, index=False)


	def get_omim_map_all_labels(self, keep_links='all'):
		"""
		Args:
			keep_links (str or set)
		Returns:
			dict: {OMIM_CODE: [dis_code1, ...]}
		"""
		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'B'}
		row_list = pd.read_csv(self.DISEASE_MAP_CSV).fillna('').to_dict(orient='records')
		ret_dict = {}
		for row_item in row_list:
			omim_code = row_item['OMIM_CODE']
			orpha_code, orpha_link = row_item['ORPHA_CODE'], row_item['ORPHA_LINK']
			ccrd_code, ccrd_link = row_item['CCRD_CODE'], row_item['CCRD_LINK']
			dis_codes = [omim_code]
			if orpha_code and orpha_link in keep_links:
				dis_codes.append(orpha_code)
			if ccrd_code and ccrd_link in keep_links:
				dis_codes.append(ccrd_code)
			ret_dict[omim_code] = dis_codes
		return ret_dict


	def get_labels_set_with_all_eq_sources(self, sources):
		"""
		Returns:
			set: {sorted_dis_codes_tuple, ...}; sorted_dis_codes_tuple = (dis_code1, dis_code2, ...)
		"""
		omim_to_dis_codes = self.get_omim_map_all_labels(keep_links={'E'})
		return set([tuple(sorted(dis_codes)) for dis_codes in omim_to_dis_codes.values() if self.diseases_from_all_sources(dis_codes, sources)])


	def gen_disease_auto_match_csv(self):
		omim_reader = OMIMReader()
		omim_to_cns_info = {omim:info['CNS_NAME'] for omim, info in omim_reader.get_cns_omim().items()}
		omim_to_eng_info = {omim:info['ENG_NAME'] for omim, info in omim_reader.get_omim_dict().items()}
		orpha_reader = OrphanetReader()
		orpha_to_cns_info = {orpha:info['CNS_NAME'] for orpha, info in orpha_reader.get_cns_orpha_dict().items()}
		orpha_to_eng_info = {orpha:info['ENG_NAME'] for orpha, info in orpha_reader.get_orpha_dict().items()}
		omim_to_exact_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'E'})
		omim_to_broad_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'NTBT'})
		hpo_reader = HPOReader()
		anno_dis = set(hpo_reader.get_dis_list())
		pid_to_pitem = self.get_pid_to_pitem()
		pa_omim_codes = [self.get_dis_code(p_item['MainData']['Diagnosis']) for pid, p_item in pid_to_pitem.items()]
		pa_omim_codes = sorted(list(set([omim_code for omim_code in pa_omim_codes if omim_code] )))
		row_infos = []
		for omim_code in pa_omim_codes:
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
		pd.DataFrame(row_infos).to_csv(self.DISEASE_MAP_AUTO_CSV, index=False,
			columns=['OMIM_CODE', 'OMIM_ENG', 'OMIM_CNS', 'ORPHA_LINK', 'ORPHA_CODE', 'ORPHA_ENG', 'ORPHA_CNS'])


	def lab_item_to_hpos(self, labItem, lab_findings_map_hpos):
		"""
		Args:
			labItem (dict): e.g. {"Name": "ALAT (GPT)", "Age": "3.29 Week(s)", "Specimen": "serum/plasma", "Quantity": "57 U/l", "Quality": "Increased"},
		Returns:
			list: [hpo_code1, hpo_code2, ...]
		"""
		return lab_findings_map_hpos.get((labItem['Name'].strip(), labItem['Specimen'].strip(), labItem['Quality'].strip()), [])


	def get_lab_finding_map_hpos(self):
		"""
		Returns:
			dict: {(name, specimen, quality): [hpo_code, ...]}
		"""
		if self.lab_finding_map_hpos:
			return self.lab_finding_map_hpos
		ret_dict = {}
		row_list = pd.read_csv(self.LAB_FINDING_MAP_HPO_CSV).fillna('').to_dict(orient='records')
		print(row_list[:10])
		for row_item in row_list:
			if len(row_item['HPO']) == 0:
				continue
			if row_item['TYPE'] in self.remove_sym_types:
				continue
			print(row_item['NAME'])
			dict_list_add((row_item['NAME'].strip(), row_item['SPECIMEN'].strip(), row_item['QUALITY'].strip()), row_item['HPO'].strip(), ret_dict)
		self.lab_finding_map_hpos = ret_dict
		return self.lab_finding_map_hpos


	def gen_lab_finding_auto_match_csv(self):
		dedup_set = set()
		pid_to_pitem = self.get_pid_to_pitem()
		data = []
		for pid, p_item in pid_to_pitem.items():
			for labDict in p_item['Lab findings']:
				row = (labDict['Name'], labDict['Specimen'], labDict['Quality'])
				if row[2] == 'Normal' or row[2] == 'Unknown' or str(row) in dedup_set:
					continue
				data.append(row)
				dedup_set.add(str(row))
		pd.DataFrame(data, columns=['NAME', 'SPECIMEN', 'QUALITY']).to_csv(self.LAB_FINDING_MAP_HPO_AUTO_CSV, index=False)


	def get_dis_code(self, diagnosis):
		"""
		Returns:
			str or None
		"""
		match_obj = re.match('^.* \(MIM (\d+)\)$', diagnosis)
		dis_code = 'OMIM:' + match_obj.group(1).strip()
		return None if dis_code == 'OMIM:1' else dis_code


	def check_dis_code(self):
		pid_to_pitem = self.get_pid_to_pitem()
		for pid, p_item in pid_to_pitem.items():
			print(self.get_dis_code(p_item['MainData']['Diagnosis']))


	def have_symptom(self, sym_range):
		return not (sym_range == 'non' or sym_range == 'no' or sym_range == '-' or sym_range == 'normal')


	def get_sym_range_type(self, symptom):
		pid_to_pitem = self.get_pid_to_pitem()
		return unique_list([symItem['Range'] for pid, p_item in pid_to_pitem.items() for symItem in p_item['Symptoms'] if symItem['Name'].strip() == symptom])


	def get_sym_age_type(self, symptom):
		pid_to_pitem = self.get_pid_to_pitem()
		return unique_list([symItem['Age'] for pid, p_item in pid_to_pitem.items() for symItem in p_item['Symptoms'] if symItem['Name'].strip() == symptom])


	def get_all_range_type(self):
		pid_to_pitem = self.get_pid_to_pitem()
		return unique_list([symItem['Range'] for pid, p_item in pid_to_pitem.items() for symItem in p_item['Symptoms']])


	def get_all_diag_confirm(self):
		pid_to_pitem = self.get_pid_to_pitem()
		return unique_list([p_item['MainData']['Diagnosis confirmed'] for pid, p_item in pid_to_pitem.items()])


	def get_all_new_born_screen(self):
		pid_to_pitem = self.get_pid_to_pitem()
		return unique_list([p_item['MainData']['Found in newborn screening'] for pid, p_item in pid_to_pitem.items()])


	def get_death_year(self, ageStr):
		match_obj = re.match('^(.*) (Day|Week|Month|Year)\(s\)$', ageStr)
		age, unit = float(match_obj.group(1).strip()), match_obj.group(2).strip()
		if unit == 'Day':
			age /= 365
		elif unit == 'Week':
			age = age*7/365
		elif unit == 'Month':
			age /= 12
		elif unit == 'Year':
			pass
		else:
			assert False
		return age


	def death_age_to_hpos(self, ageStr):
		age = self.get_death_year(ageStr)
		return [hpo for hpo, yr in self.hpo_to_death_year_range.items() if age >= yr[0] and age < yr[1]]





if __name__ == '__main__':
	pass
	from core.explainer.dataset_explainer import LabeledDatasetExplainer
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])  # HPOReader()
	rpg = RamedisPatientGenerator(hpo_reader)

	patients = rpg.get_patients()    # 767
	explainer = LabeledDatasetExplainer(patients)
	json.dump(explainer.explain(), open(os.path.join(rpg.OUTPUT_PATIENT_FOLDER, 'patients_statistics.json'), 'w'), indent=2, ensure_ascii=False)
	simple_dist_plot(os.path.join(rpg.OUTPUT_PATIENT_FOLDER, 'patients_hpo.jpg'), [len(hpo_list) for hpo_list, _ in patients], 30, 'HPO Number of Patient')

	patients = rpg.get_diag_patients()    # 683
	explainer = LabeledDatasetExplainer(patients)
	json.dump(explainer.explain(), open(os.path.join(rpg.OUTPUT_PATIENT_FOLDER, 'diag_patients_statistics.json'), 'w'), indent=2, ensure_ascii=False)
	simple_dist_plot(rpg.OUTPUT_PATIENT_FOLDER + '/diag_patients_hpo.jpg', [len(hpo_list) for hpo_list, _ in patients], 30, 'HPO Number of Patient')

	patients = rpg.get_diag_not_screen_patients()   # 440
	explainer = LabeledDatasetExplainer(patients)
	json.dump(explainer.explain(), open(os.path.join(rpg.OUTPUT_PATIENT_FOLDER, 'diag_notscreen_patients_statistics.json'), 'w'), indent=2, ensure_ascii=False)
	simple_dist_plot(rpg.OUTPUT_PATIENT_FOLDER + '/diag_notscreen_patients_hpo.jpg', [len(hpo_list) for hpo_list, _ in patients], 30, 'HPO Number of Patient')



