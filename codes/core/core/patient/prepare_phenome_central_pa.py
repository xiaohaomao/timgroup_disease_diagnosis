

import json
import os
import numpy as np

from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import get_file_list, check_load_save
from core.patient.patient_generator import PatientGenerator


class PCPatientGenerator(PatientGenerator):
	def __init__(self):
		super(PCPatientGenerator, self).__init__()
		self.PHENOTIPS_PUBLIC_JSON = DATA_PATH+'/raw/PhenomeCentral/phenotips_2018-09-14_01-10.json'
		self.SIMILAR_CASE_FOLDER = DATA_PATH+'/raw/PhenomeCentral/SimilarCase'

		self.OUTPUT_PATIENT_FOLDER = DATA_PATH+'/preprocess/patient/PHENOME_CENTRAL'
		os.makedirs(self.OUTPUT_PATIENT_FOLDER, exist_ok=True)
		self.PC_PATIENTS_JSON = self.OUTPUT_PATIENT_FOLDER + '/PCPatients.json'
		self.patients = None


	@check_load_save('patients', 'PC_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_patients(self):
		pid_set = set()
		pa_list = self.get_pa_list_from_public(pid_set)
		pa_list.extend(self.get_pa_list_from_similar_case(pid_set))    # [[hpo_list, dis_list], ...]
		pa_list = self.filter_pa_list_with_exist_dis(pa_list)  # make sure diagnosis exists
		pa_list = self.filter_pa_list_with_process_hpo_list(pa_list)    # old -> new; redundancy
		return pa_list


	def statistic(self):
		pa_list = self.get_patients()

		hpo_nums = list(map(lambda pa:len(pa[0]), pa_list))


	def dis_id_to_standard(self, dis_id):
		if dis_id.startswith('ORDO'):
			return 'ORPHA:' + dis_id.split(':').pop()
		if dis_id.startswith('MIM'):
			return 'OMIM:' + dis_id.split(':').pop()
		assert False


	def get_pa_list_from_info_list(self, info_list, pid_set, pid_key='report_id'):
		"""
		Returns:
			list: [[hpo_list, dis_list], ...]
		"""
		pa_list = []
		for info_dict in info_list:
			pid = info_dict[pid_key]
			if pid in pid_set:
				continue
			pid_set.add(pid)
			disorders = [self.dis_id_to_standard(dis_infoDict['id']) for dis_infoDict in info_dict.get('disorders', []) if dis_infoDict.get('id', '').strip() != ''] # 疾病id不为空
			if len(disorders) == 0:
				continue
			hpo_list = [hpoInfoDict['id'] for hpoInfoDict in info_dict['features'] if hpoInfoDict['observed'] == 'yes' and 'id' in hpoInfoDict]   # 症状观察到
			pa_list.append([hpo_list, disorders])
		return pa_list


	def get_pa_list_from_public(self, pid_set):
		info_list = json.load(open(DATA_PATH+'/raw/PhenomeCentral/phenotips_2018-09-14_01-10.json'))
		return self.get_pa_list_from_info_list(info_list, pid_set, pid_key='report_id')


	def get_pa_list_from_similar_case(self, pid_set):
		file_list = get_file_list(DATA_PATH+'/raw/PhenomeCentral/SimilarCase', lambda file_path: file_path.split('.').pop() == 'json')
		pa_list = []
		for json_path in file_list:

			info_list = json.load(open(json_path))['results']
			pa_list.extend(self.get_pa_list_from_info_list(info_list, pid_set, pid_key='id'))
		return pa_list



if __name__ == '__main__':
	pcpg = PCPatientGenerator()
	pcpg.get_patients()











