

import json
import os

from core.utils.constant import DATA_PATH, LOG_PATH, JSON_FILE_FORMAT
from core.script.spider.get_decipher import ALL_PATIENT_PHENOTYPES, ALL_PATIENT_SNVs, ALL_PATIENT_CNVs
from core.utils.utils import dict_list_add, get_logger, slice_list_with_keep_func, delete_redundacy, check_load_save, reverse_dict_list, split_path
from core.patient.patient_generator import PatientGenerator
from core.explainer.dataset_explainer import LabeledDatasetExplainer


class DecipherPatientGenerator(PatientGenerator):
	def __init__(self):
		super(DecipherPatientGenerator, self).__init__()
		self.OUTPUT_PATIENT_FOLDER = DATA_PATH + '/preprocess/patient/DICIPHER'

		self.DECIPHER_DISEASE_PATIENTS_JSON = self.OUTPUT_PATIENT_FOLDER + '/DecipherHPORedundacyDiseasePatients.json'
		self.snv_patients = None
		self.DECIPHER_SINGLE_SNV_DISEASE_PATIENTS_JSON = self.OUTPUT_PATIENT_FOLDER + '/DecipherHPORedundacySingleSNVDiseasePatients.json'
		self.one_snv_patients = None
		self.DECIPHER_SINGLE_SNV_SINGLE_DISEASE_PATIENTS_JSON = self.OUTPUT_PATIENT_FOLDER + '/DecipherHPORedundacySingleSNVSingleDiseasePatients.json'
		self.one_snv_one_dis_patients = None
		self.DECIPHER_CNV_DISEASE_PATIENTS_JSON = self.OUTPUT_PATIENT_FOLDER + '/CNVPatients.json'
		self.cnv_patients = None

		self.snv_data_dict = None
		self.cnv_data_dict = None


	def get_patients(self):
		return self.get_snv_patients()


	def get_upatients(self):
		return self.get_cnv_upatients()


	@check_load_save('snv_patients', 'DECIPHER_DISEASE_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_snv_patients(self):
		"""
		"""
		datapath = self.DECIPHER_DISEASE_PATIENTS_JSON
		logpath = datapath[:-5] + '-LOG'
		logger = get_logger('SnvPatients', log_path=logpath, mode='w')
		pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list = self.before_gen_snv_patients(logger)
		pa_list = self.generate_disease_patients(pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list)
		pa_list = self.filter_pa_list_with_exist_dis(pa_list)
		self.statistic(pa_list, split_path(datapath)[1], logger)
		return pa_list


	@check_load_save('one_snv_patients', 'DECIPHER_SINGLE_SNV_DISEASE_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_one_snv_patients(self):
		"""
		"""
		datapath = self.DECIPHER_SINGLE_SNV_DISEASE_PATIENTS_JSON
		logpath = datapath[:-5] + '-LOG'
		logger = get_logger('SnvPatients', log_path=logpath, mode='w')
		pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list = self.before_gen_snv_patients()
		pid_to_gene_symbols = {pid:geneSymbols for pid, geneSymbols in pid_to_gene_symbols.items() if len(geneSymbols) == 1}
		pa_list = self.generate_disease_patients(pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list)
		pa_list = self.filter_pa_list_with_exist_dis(pa_list)
		self.statistic(pa_list, split_path(datapath)[1], logger)
		return pa_list


	@check_load_save('one_snv_one_dis_patients', 'DECIPHER_SINGLE_SNV_SINGLE_DISEASE_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_one_snv_one_dis_patients(self):

		datapath = self.DECIPHER_SINGLE_SNV_SINGLE_DISEASE_PATIENTS_JSON
		logpath = datapath[:-5] + '-LOG'
		logger = get_logger('SnvPatients', log_path=logpath, mode='w')
		pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list = self.before_gen_snv_patients()
		pid_to_gene_symbols = {pid:geneSymbols for pid, geneSymbols in pid_to_gene_symbols.items() if len(geneSymbols) == 1}
		pa_list = self.generate_disease_patients(pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list)
		pa_list = [[hpo_list, dis_list] for hpo_list, dis_list in pa_list if len(dis_list) == 1 and dis_list[0] in self.all_dis_set]
		self.statistic(pa_list, split_path(datapath)[1], logger)
		return pa_list


	@check_load_save('cnv_patients', 'DECIPHER_CNV_DISEASE_PATIENTS_JSON', JSON_FILE_FORMAT)
	def get_cnv_upatients(self, snv_overlap=False):
		"""
		Returns:
			list: [upatient, ...], upatient=[hpo1, hpo2]
		"""
		pid_to_hpos = self.get_pid_to_hpos()
		intersect_pids, only_snv_pids, only_cnv_pids = dpg.get_csnv_intersect()
		pid_set = (only_cnv_pids|intersect_pids) if snv_overlap else only_cnv_pids
		return [pid_to_hpos[pid] for pid in pid_set if len(pid_to_hpos[pid]) > 0]


	def get_all_upatients(self):
		pid_to_hpos = self.get_pid_to_hpos()
		return [hpo_list for pid, hpo_list in pid_to_hpos.items() if len(hpo_list) > 0]


	def generate_disease_patients(self, pid_to_hpos, pid_to_gene_symbols, geneSymbolToDis):
		"""
		Returns:
			list: [[hpo_list, diseaseList], ...]
		"""
		pa_list = []
		for pid in set(pid_to_hpos.keys()) & set(pid_to_gene_symbols.keys()):
			dis_set = set()
			for geneSymbol in pid_to_gene_symbols[pid]:
				dis_set.update(geneSymbolToDis.get(geneSymbol, []))
			if len(dis_set) == 0:
				continue
			pa_list.append([pid_to_hpos[pid], list(dis_set)])
		return pa_list


	def get_pid_to_hpos(self, logger=None):
		pid_to_hpos = json.load(open(ALL_PATIENT_PHENOTYPES))
		if logger is None:
			return {pid: self.process_pa_hpo_list(hpo_list) for pid, hpo_list in pid_to_hpos.items()}

		# get phenotypes; old HPO -> new HPO
		hpo_dict = self.hpo_reader.get_hpo_dict()
		old_map_new_hpo = self.hpo_reader.get_old_map_new_hpo_dict()
		for pid in pid_to_hpos:
			hpo_list = pid_to_hpos[pid]
			new_hpo_list = []
			for hpo_code in hpo_list:
				if hpo_code not in hpo_dict:
					if hpo_code in old_map_new_hpo:
						new_code = old_map_new_hpo[hpo_code]
						new_hpo_list.append(new_code)
						logger.info('pid:{0}: {1}(old) -> {2}(new)'.format(pid, hpo_code, new_code))  #
					else:
						logger.info('pid:{0}: delete {1}'.format(pid, hpo_code))  #
				else:
					new_hpo_list.append(hpo_code)
			pid_to_hpos[pid] = new_hpo_list

		# phenotypes redundacy
		redundacy_count = 0
		for pid, hpo_list in pid_to_hpos.items():
			new_hpo_list = delete_redundacy(hpo_list, hpo_dict)
			if len(new_hpo_list) != len(hpo_list):
				pid_to_hpos[pid] = new_hpo_list
				logger.info(
					'pid:{0}, HPO Terms Redundacy: {1}->{2}'.format(pid, len(hpo_list), len(new_hpo_list)))
				redundacy_count += 1
		logger.info('Redundacy Patients Number/Total Patients Number: {0}/{1} = {2}'.format(
			redundacy_count, len(pid_to_hpos), 100 * redundacy_count / len(pid_to_hpos)))

		return pid_to_hpos


	def before_gen_snv_patients(self, logger=None):
		pid_to_hpos, geneSymbolTodis_list = self.get_pid_to_hpos(logger=None), self.hpo_reader.get_gene_symbol_to_dis_list()
		_, only_snv_pids, _ = dpg.get_csnv_intersect()
		# get snv
		snv_data_dict = self.get_snv_data_dict()  # {pid: [snv_dict1, ..]}
		pid_to_gene_symbols = {pid:[snv_dict['GENE_NAME'] for snv_dict in snv_list] for pid, snv_list in snv_data_dict.items() if pid in only_snv_pids}  # {pid: [geneSymbol1, ...]}
		return pid_to_hpos, pid_to_gene_symbols, geneSymbolTodis_list


	def get_csnv_intersect(self):
		"""
		Returns:
			set: intersect_pids
			set: only_snv_pids
			set: only_cnv_pids
		"""
		snv_data_dict = self.get_snv_data_dict()
		snv_pids = {pid for pid, snv_list in snv_data_dict.items() if len(snv_list) > 0}
		cnv_data_dict = self.get_cnv_data_dict()  # {pid: [cnv_dict1, ..]}
		cnv_pids = {pid for pid, cnv_list in cnv_data_dict.items() if len(cnv_list) > 0}
		intersect_pids = snv_pids & cnv_pids   # 101
		only_snv_pids = snv_pids - intersect_pids
		only_cnv_pids = cnv_pids - intersect_pids
		return intersect_pids, only_snv_pids, only_cnv_pids


	def get_snv_data_dict(self):
		if self.snv_data_dict is None:
			self.snv_data_dict = json.load(open(ALL_PATIENT_SNVs))
		return self.snv_data_dict


	def get_cnv_data_dict(self):
		if self.cnv_data_dict is None:
			self.cnv_data_dict = json.load(open(ALL_PATIENT_CNVs))
		return self.cnv_data_dict


	def statistic(self, patients, datasetName, logger):
		from collections import Counter
		logger.info('\n{0}--------'.format(datasetName))
		logger.info('Patients Number: {0}'.format(len(patients)))
		counter = Counter()
		for hpo_list, _ in patients:
			counter[len(hpo_list)] += 1
		for hpo_num, count in sorted(counter.items()):
			logger.info('Patients with {0} HPO Terms: {1}, {2}%'.format(hpo_num, count, 100*count/len(patients)))
		logger.info('Average HPO Number per Patient: {0}'.format(sum(map(lambda item: item[0]*item[1], counter.items()))/len(patients)))

		counter = Counter()
		for _, dis_list in patients:
			counter[len(dis_list)] += 1
		for dis_num, count in sorted(counter.items()):
			logger.info('Patients with {0} Disease Terms: {1}, {2}%'.format(dis_num, count, 100*count/len(patients)))
		logger.info('Average Disease Number per patient: {0}'.format(sum(map(lambda item: item[0]*item[1], counter.items()))/len(patients)))

		explainer = LabeledDatasetExplainer(patients)
		logger.info(json.dumps(explainer.explain(), indent=2))
		simple_dist_plot(self.OUTPUT_PATIENT_FOLDER + '/{}-HPO-NUM.jpg'.format(datasetName), [len(hpo_list) for hpo_list, _ in patients], 30, 'HPO Number of Patient')
		simple_dist_plot(self.OUTPUT_PATIENT_FOLDER + '/{}-DIS-NUM.jpg'.format(datasetName), [len(dis_list) for _, dis_list in patients], 30, 'DIS Number of Patient')


if __name__ == '__main__':
	from core.explainer.dataset_explainer import UnLabeledDatasetExplainer
	from core.draw.simpledraw import simple_dist_plot
	dpg = DecipherPatientGenerator()

	upatients = dpg.get_upatients()
