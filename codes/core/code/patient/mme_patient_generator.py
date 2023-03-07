

import os
import json
from collections import Counter
import numpy as np
import pandas as pd

from core.patient.patient_generator import PatientGenerator
from core.reader import HPOReader, OMIMReader, OrphanetReader
from core.utils.constant import DATA_PATH
from core.utils.utils import delete_redundacy, check_load, JSON_FILE_FORMAT


class MMEPatientGenerator(PatientGenerator):
	def __init__(self, hpo_reader=HPOReader()):
		super(MMEPatientGenerator, self).__init__(hpo_reader)
		self.BENCHMARK_PATIENT_JSON = os.path.join(DATA_PATH, 'raw', 'MME', 'benchmark_patients.json')
		self.DISEASE_MAP_AUTO_CSV = os.path.join(DATA_PATH, 'raw', 'MME', 'disease_map_auto.csv')
		self.DISEASE_MAP_CSV = os.path.join(DATA_PATH, 'raw', 'MME', 'disease_map.csv')

		self.OUTPUT_PATIENT_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'MME')
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)

		self.MME_PIDS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'MME_pids.json')
		self.MME_PATIENTS_JSON = os.path.join(self.OUTPUT_PATIENT_FOLDER, 'MME_patients.json')
		self.patients = None
		self.pids = None


	def id2omim(self, id):
		return 'O' + id


	@check_load('patients', 'MME_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_patients(self):
		patients, pids = self._gen_patients()
		return patients


	@check_load('pids', 'MME_PIDS_JSON', JSON_FILE_FORMAT)
	def get_pids(self):
		patients, pids = self._gen_patients()
		return pids


	def _gen_patients(self):
		patients, pids = [], []
		omim_to_discodes = self.get_omim_map_all_labels()
		mme_list = json.load(open(self.BENCHMARK_PATIENT_JSON))
		for mme_pa_dict in mme_list:
			assert len(mme_pa_dict['disorders']) < 2
			if len(mme_pa_dict['disorders']) == 0:
				dis_codes = []
			else:
				dis_code =  self.id2omim(mme_pa_dict['disorders'][0]['id'])
				dis_codes = omim_to_discodes[dis_code]
			hpo_list = []
			for feature_dict in mme_pa_dict['features']:
				if feature_dict['observed'] == 'no':
					continue
				hpo_list.append(feature_dict['id'])
			hpo_list = self.process_pa_hpo_list(hpo_list, reduce=False)
			patients.append([hpo_list, dis_codes])
			pids.append(mme_pa_dict['id'])
		json.dump(patients, open(self.MME_PATIENTS_JSON, 'w'), indent=2)
		json.dump(pids, open(self.MME_PIDS_JSON, 'w'), indent=2)
		return patients, pids


	def gen_disease_auto_match_csv(self):
		omim_reader = OMIMReader()
		omim_to_cns_info = {omim:info['CNS_NAME'] for omim, info in omim_reader.get_cns_omim().items()}
		omim_to_eng_info = {omim:info['ENG_NAME'] for omim, info in omim_reader.get_omim_dict().items()}
		orpha_reader = OrphanetReader()
		orpha_to_cns_info = {orpha:info['CNS_NAME'] for orpha, info in orpha_reader.get_cns_orpha_dict().items()}
		orpha_to_eng_info = {orpha:info['ENG_NAME'] for orpha, info in orpha_reader.get_orpha_dict().items()}
		omim_to_exact_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'E'})
		omim_to_broad_orpha = orpha_reader.get_omim_to_orpha_list(keep_links={'NTBT'})

		mme_patients = json.load(open(self.BENCHMARK_PATIENT_JSON))
		pa_omim_codes = [self.id2omim(pa_dict['disorders'][0]['id']) for pa_dict in mme_patients if pa_dict['disorders']]
		pa_omim_codes = sorted(list(set(pa_omim_codes)))

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


	def statistic(self):
		patients = self.get_patients()
		print('patients Number =', len(patients))  # 43
		hpo_nums = list(map(lambda pa:len(pa[0]), patients))
		print('average hpo_num = ', np.mean(hpo_nums))  # 11.581395348837209
		print('max hpo_num = ', max(hpo_nums))  # 26
		print('min hpo_num = ', min(hpo_nums))  # 2


if __name__ == '__main__':
	from core.draw.simpledraw import simple_dist_plot
	from core.explainer.dataset_explainer import LabeledDatasetExplainer
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA'])  # HPOReader()
	mme_pg = MMEPatientGenerator(hpo_reader)

	patients = mme_pg.get_patients()
	explainer = LabeledDatasetExplainer(patients)
	json.dump(explainer.explain(), open(os.path.join(mme_pg.OUTPUT_PATIENT_FOLDER, 'patients_statistics.json'), 'w'),indent=2, ensure_ascii=False)
	simple_dist_plot(os.path.join(mme_pg.OUTPUT_PATIENT_FOLDER, 'patients_hpo.jpg'), [len(hpo_list) for hpo_list, _ in patients], 30, 'HPO Number of Patient')






