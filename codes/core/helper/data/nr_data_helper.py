


import json
import os
import random

from core.utils.constant import DATA_PATH
from core.utils.constant import GENE_ANNOTATION, DATASET_TYPE_M, DATASET_TYPE_I, DATASET_TYPE_S, DATASET_TYPE_O
from core.utils.utils import ret_same
from core.predict.PageRankNoiseReductor import PageRankNoiseReductor
from statistician.statcommon import get_no_noise_dataset


class NoiseReduceDataHelper(object):
	def __init__(self):
		self.data_noise_reduct_paths = {
			'PagerankNoiseReductor':{
				'SIM_ORIGIN':DATA_PATH + '/preprocess/SIMULATION/OriginPGNR.json',
				'SIM_NOISE':DATA_PATH + '/preprocess/SIMULATION/NoisePGNR.json',
				'SIM_IMPRE':DATA_PATH + '/preprocess/SIMULATION/ImprecisionPGNR.json',
				'SIM_IMPRE_NOISE':DATA_PATH + '/preprocess/SIMULATION/ImpreNoisePGNR.json',
				'SIM_NOISE_IMPRE':DATA_PATH + '/preprocess/SIMULATION/NoiseImprePGNR.json',
				'MMEcalculating Metrics_43':DATA_PATH + '/preprocess/MME/MMEPatientsPGNR.json',
				'DEC_3236':DATA_PATH + '/preprocess/DICIPHER/DecipherHPORedundacyDiseasePatientsPGNR.json',
				'DEC_SNV_2779':DATA_PATH + '/preprocess/DICIPHER/DecipherHPORedundacySingleSNVDiseasePatientsPGNR.json',
				'DEC_SNV_DIS_155':DATA_PATH + '/preprocess/DICIPHER/DecipherHPORedundacySingleSNVSingleDiseasePatientsPGNR.json',
				'PC_174':DATA_PATH + '/preprocess/PHENOME_CENTRAL/PCPatientsPGNR.json',
			}
		}
		pass


	def get_no_noise_dataset_name(self, originName, keep_types):
		type_to_mark = {DATASET_TYPE_M:'M', DATASET_TYPE_I:'I', DATASET_TYPE_S:'S', DATASET_TYPE_O:'O'}
		return '{}_{}'.format(originName, ''.join([type_to_mark[type] for type in sorted(keep_types)]))


	def load_no_noise_test_data(self, keep_types, data_names=None):
		"""
		Args:
			keep_types
		"""

		def fill_empty(new_dataset, rawDataset):
			assert len(new_dataset) == len(rawDataset)
			for i in range(len(new_dataset)):
				if len(new_dataset[i][0]) == 0:
					new_dataset[i][0] = random.sample(rawDataset[i][0], 1)

		self.load_test_data(data_names)
		data = {}
		for data_name, dataset in self.data.items():
			name = self.get_no_noise_dataset_name(data_name, keep_types)
			new_dataset = get_no_noise_dataset(dataset, keep_types)
			fill_empty(new_dataset, dataset)
			data[name] = new_dataset
		self.data = data

	def load_test_data(self, data_names=None, noise_reductor=None, keep_k_func=ret_same):
		if data_names is None:
			data_names = self.data_names
		if noise_reductor is None:
			for data_name in data_names:
				self.data[data_name] = json.load(open(self.dataPaths[data_name]))
		else:
			for data_name in data_names:
				self.data[data_name] = self.get_noise_reduct_data(data_name, noise_reductor, keep_k_func)
		self.changeDisCodeToList()


	def get_noise_reduct_data(self, data_name, noise_reductor, keep_k_func=ret_same):
		testdata = json.load(open(self.dataPaths[data_name]))
		json_path = self.data_noise_reduct_paths[noise_reductor.name][data_name]
		if os.path.exists(json_path):
			phe_lists = json.load(open(json_path))
		else:
			pg_reductor = PageRankNoiseReductor()
			phe_lists, _ = pg_reductor.reduct_noise_for_many([phe_list for phe_list, _ in testdata], order=True,
			                                            anno_used=GENE_ANNOTATION)
			json.dump(phe_lists, open(json_path, 'w'), indent=2)
		phe_lists = [phe_list[:keep_k_func(len(phe_list))] for phe_list in phe_lists]
		for i in range(len(testdata)):
			testdata[i][0] = phe_lists[i]
		return testdata



