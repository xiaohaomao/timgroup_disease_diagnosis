

import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import itertools

from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL, SEED, VALIDATION_TEST_DATA
from core.utils.constant import DATA_PATH, PHELIST_ANCESTOR, PHELIST_REDUCE, PHELIST_ANCESTOR_DUP, JSON_FILE_FORMAT, VEC_TYPE_IDF, ROOT_HPO_CODE
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF, VEC_TYPE_TF_IDF, TEST_DATA, VALIDATION_DATA, VEC_TYPE_PROB, VEC_TYPE_LOG_PROB
from core.utils.utils import item_list_to_rank_list, get_save_func, data_to_01_matrix, data_to_tf_dense_matrix, data_to_tf_matrix, data_to_01_dense_matrix
from core.utils.utils import get_all_ancestors_for_many, delete_redundacy, get_all_dup_ancestors_for_many, del_obj_list_dup, count_obj_list_dup
from core.utils.utils import get_all_ancestors_for_many_with_ances_dict, delete_redundacy_with_ances_dict, get_all_dup_ancestors_for_many_with_ances_dict
from core.utils.utils import slice_list_with_keep_set, check_return, split_path, get_all_ancestors, get_all_descendents, unique_list
from core.utils.utils import combine_key_to_list
from core.reader import HPOReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader, RDReader, RDFilterReader, source_codes_to_rd_codes
from core.helper.data.base_data_helper import BaseDataHelper
from core.explainer.dataset_explainer import LabeledDatasetExplainer
from core.patient import CJFHPatientGenerator, RamedisPatientGenerator, MMEPatientGenerator, PUMCPatientGenerator, PatientGenerator, ThanPatientGenerator, HmsPatientGenerator


class DataHelper(BaseDataHelper):
	def __init__(self, hpo_reader=None, rd_reader=None):
		super(DataHelper, self).__init__()
		integrate_prefix = 'INTEGRATE_'
		self.hpo_reader = hpo_reader or HPOReader()
		self.rd_reader = rd_reader or RDReader()

		self.use_rd_code = self.hpo_reader.name.startswith(integrate_prefix)
		#self.use_rd_code = True
		self.all_dis_set = None
		self.tf_idf_transformer = None

		self.TEST_VAL_STATISTIC_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test_val_statistics')
		os.makedirs(self.TEST_VAL_STATISTIC_FOLDER, exist_ok=True)

		self.TEST_SIZE = 0.6
		self.VALID_TEST_MIN_HPO_NUM = 3
		self.default_sources = None

		self.keep_dis_to_remove = {
			# RAMEDIS
			'OMIM:125850': ['ORPHA:552'],
			'OMIM:210200': ['ORPHA:6'],
			'OMIM:166200': ['CCRD:86'],
			'OMIM:176000': ['CCRD:92'],
			'OMIM:215700': ['CCRD:18'],
			'OMIM:251000': ['CCRD:71'],
			'OMIM:251100': ['CCRD:71'],
			'OMIM:251110': ['CCRD:71'],
			'OMIM:261630': ['CCRD:49'],
			'OMIM:277400': ['CCRD:71'],
			'OMIM:277440': ['CCRD:51'],
			'OMIM:307800': ['CCRD:51'],
			'OMIM:309900': ['CCRD:73'],
			'OMIM:605814': ['CCRD:18'],

			# MME
			'OMIM:604317': ['ORPHA:2512'],
			'OMIM:612541': ['CCRD:104'],
		}

		self.keep_dis_to_remove = combine_key_to_list(self.keep_dis_to_remove, HmsPatientGenerator().get_keep_to_general_dis())

		self.dataset_mark = self.hpo_reader.name[len(integrate_prefix):] if self.use_rd_code else self.hpo_reader.name




		self.origin_to_path = {
			# labeled data
			'SIM_ORIGIN': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'SIMULATION', 'origin.json'), # 4400
			'SIM_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'SIMULATION', 'noise.json'), # 4400
			'SIM_IMPRE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'SIMULATION', 'imprecision.json'), # 4400
			'SIM_IMPRE_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'SIMULATION', 'impre_noise.json'), # 4400
			'SIM_NOISE_IMPRE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'SIMULATION', 'noise_impre.json'), # 4400

		}

		self.test_names = [

			# validation subset of RAMEDIS
			'Validation_subsets_of_RAMEDIS',
			# Multi-country-test set
			'Multi-country-test',

			# combined multi-country set
			'Combined-Multi-Country',

			
			# PUMCH-L datasest
			'PUMCH-L-CText2Hpo',
			'PUMCH-L-Meta',
			'PUMCH-L-CHPO',

			# PUMCH-MDT dataset
			'PUMCH-MDT',

			# PUMCH-ADM dataset
			'PUMCH-ADM',

			# Sampled_100 cases
			'Multi-country-test-set-100',
			'RAMEDIS_100',

			# 24 methylmalonic academia cases  using different knowledge bases
			'MUT_24_CASES_OMIM',
			'MUT_24_CASES_ORPHA',
			'MUT_24_CASES_CCRD',
			'MUT_24_CASES_OMIM_ORPHA',
			'MUT_24_CASES_CCRD_ORPHA',
			'MUT_24_CASES_CCRD_OMIM',
			'MUT_24_CASES_CCRD_OMIM_ORPHA',

			# validation subsets of RAMEDIS using different knowledge bases
			'validation_subset_RAMDEIS_CCRD',
			'validation_subset_RAMDEIS_OMIM',
			'validation_subset_RAMDEIS_ORPHA',
			'validation_subset_RAMDEIS_CCRD_OMIM',
			'validation_subset_RAMDEIS_CCRD_ORPHA',
			'validation_subset_RAMDEIS_OMIM_ORPHA',
			'validation_subset_RAMDEIS_CCRD_OMIM_ORPHA',

			# multi_country_test using different knowledge bases
			'Multi-country-test_CCRD',
			'Multi-country-test_OMIM',
			'Multi-country-test_ORPHA',
			'Multi-country-test_CCRD_OMIM',
			'Multi-country-test_CCRD_ORPHA',
			'Multi-country-test_OMIM_ORPHA',
			'Multi-country-test_CCRD_OMIM_ORPHA',


			# simulated datasets
			'SIM_ORIGIN',
			'SIM_NOISE',
			'SIM_IMPRE',
			'SIM_IMPRE_NOISE',

		]

		self.test_to_path = {

			# Validation subset of RAMEDIS
			'Validation_subsets_of_RAMEDIS': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"Validation_subsets_of_RAMEDIS.json"),

			# Multi-country set dataset
			'Multi-country-test': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test', "Multi-country-test.json"),

			# Combined Multi-Country dataset
			'Combined-Multi-Country': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"Combined-Multi-Country.json"),

			# PUMCH-L dataset
			'PUMCH-L-CText2Hpo':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L.json"),

			# PUMCH-MDT dataset
			'PUMCH-MDT': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"PUMCH-MDT.json"),

			# PUMCH-ADM dataset
			'PUMCH-ADM': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"PUMCH-ADM.json"),


			# PUMCH-L dataset based on Meta Thesaurus phenotype extraction method
			'PUMCH-L-Meta': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L-Meta.json"),

			# PUMCH-L dataset based on CHPO phenotype extraction method
			'PUMCH-L-CHPO': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L-CHPO.json"),

			# sampled 100 cases
			'Multi-country-test-set-100':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test-set-SAMPLE_100.json'),
			'RAMEDIS_100':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'RAMEDIS_SAMPLE_100.json'),

			#
			# 24 methylmalonic academia cases  using different knowledge bases
			'MUT_24_CASES_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_OMIM.json'),
			'MUT_24_CASES_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_ORPHA.json'),
			'MUT_24_CASES_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD.json'),
			'MUT_24_CASES_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_OMIM_ORPHA.json'),
			'MUT_24_CASES_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_ORPHA.json'),
			'MUT_24_CASES_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_OMIM.json'),
			'MUT_24_CASES_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_OMIM_ORPHA.json'),



			# validation subsets of RAMEDIS using different knowledge bases
			'Multi-country-tuning_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD.json'),
			'Multi-country-tuning_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_OMIM.json'),
			'Multi-country-tuning_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_ORPHA.json'),
			'Multi-country-tuning_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_OMIM.json'),
			'Multi-country-tuning_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_ORPHA.json'),
			'Multi-country-tuning_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_OMIM_ORPHA.json'),
			'Multi-country-tuning_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_OMIM_ORPHA.json'),


			#multi_country_test  using different knowledge bases
			'Multi-country-test_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD.json'),
			'Multi-country-test_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_OMIM.json'),
			'Multi-country-test_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_ORPHA.json'),
			'Multi-country-test_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_OMIM.json'),
			'Multi-country-test_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_ORPHA.json'),
			'Multi-country-test_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_OMIM_ORPHA.json'),
			'Multi-country-test_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_OMIM_ORPHA.json'),


			# simulated datasets
			'SIM_ORIGIN': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_ORIGIN.json'),
			'SIM_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_NOISE.json'),
			'SIM_IMPRE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_IMPRE.json'),
			'SIM_IMPRE_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_IMPRE_NOISE.json'),


		}






	@check_return('default_sources')
	def get_default_sources(self):
		dis_list = self.hpo_reader.get_dis_list()
		sources = set()
		for dis_code in dis_list:
			sources.add(dis_code.split(':')[0])
		return sources


	def gen_complete_lb_test_patients(self, dnames):
		"""
		"""
		sources = {'OMIM', 'ORPHA', 'CCRD'}
		hpo_reader = HPOFilterDatasetReader(keep_dnames=sources)
		for dname in dnames:
			new_dname = dname+'_COMP_LB'
			sources = sources or self.get_default_sources()
			save_json = self.test_to_path[new_dname]
			if dname == 'RAMEDIS':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', 'RAMEDIS.json'),)
				qualifed_dis_codes = RamedisPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			elif dname == 'MME':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', 'MME.json'), )
				qualifed_dis_codes = MMEPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			elif dname == 'CJFH':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', 'CJFH.json'),)
				qualifed_dis_codes = CJFHPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			elif dname == 'PUMC':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', "PUMC-['入院情况', '出院诊断', '既往史', '现病史']-del_diag_hpo_False.json"),)
				patients = [p for p in patients if 'OMIM:148000' not in p[1]]   # OMIM:148000
				qualifed_dis_codes = PUMCPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			elif dname == 'THAN':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', 'THAN.json'),)
				qualifed_dis_codes = ThanPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			elif dname == 'HMS':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, 'test', 'HMS.json'), )
				qualifed_dis_codes = HmsPatientGenerator(hpo_reader=hpo_reader).get_labels_set_with_all_eq_sources(sources)
			else:
				raise RuntimeError('Wrong dataset name: {}'.format(dname))
			pg = PatientGenerator(hpo_reader=hpo_reader)
			filtered_qualifed_dis_codes = set()
			for dis_tuple in qualifed_dis_codes:
				dis_tuple = tuple(sorted(pg.process_pa_dis_list(dis_tuple)))
				if pg.diseases_from_all_sources(dis_tuple, sources):
					filtered_qualifed_dis_codes.add(dis_tuple)
			patients = [p for p in patients if tuple(sorted(p[1])) in filtered_qualifed_dis_codes]
			if len(patients) == 0:
				print('{}: empty'.format(dname))
			else:
				patients = PatientGenerator(hpo_reader=self.hpo_reader).filter_pa_list_with_exist_dis(patients)
				os.makedirs(os.path.dirname(save_json), exist_ok=True)
				json.dump(patients, open(save_json, 'w'), indent=2)
				LabeledDatasetExplainer(patients).explain_save_json(os.path.join(self.TEST_VAL_STATISTIC_FOLDER, f'{new_dname}.json'))


	def gen_selected_sources_patients(self, dnames):
		all_sources = {'OMIM', 'ORPHA', 'CCRD'}
		hpo_reader = HPOFilterDatasetReader(keep_dnames=all_sources)
		eval_datas = [TEST_DATA, VALIDATION_DATA]
		for dname, eval_data in itertools.product(dnames, eval_datas):
			save_json = self.test_to_path.get(dname, '') if eval_data == TEST_DATA else self.valid_to_path.get(dname, '')
			folder_mark = 'test' if eval_data == TEST_DATA else 'validation'
			if not save_json:
				continue
			if dname == 'RAMEDIS':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, 'RAMEDIS.json'), )
			elif dname == 'MME':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, 'MME.json'), )
			elif dname == 'CJFH':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, 'CJFH.json'), )
			elif dname == 'PUMC':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, "PUMC-['入院情况', '出院诊断', '既往史', '现病史']-del_diag_hpo_False.json"), )
			elif dname == 'THAN':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, 'THAN.json'), )
			elif dname == 'HMS':
				patients = self.get_dataset_with_path(os.path.join(DATA_PATH, 'preprocess', 'patient', hpo_reader.name, folder_mark, 'HMS.json'), )
				pass
			else:
				raise RuntimeError('Wrong dataset name: {}'.format(dname))
			patients = self.filter_patients(patients)
			os.makedirs(os.path.dirname(save_json), exist_ok=True)
			json.dump(patients, open(save_json, 'w'), indent=2)
		self.multi_dataset_statistics(dnames)


	def dis_filter(self, dataset):
		all_dis_set = self.get_all_dis_set()
		ret = []
		for p_hpo_list, pdis_list in dataset:
			newpdis_list = slice_list_with_keep_set(pdis_list, all_dis_set)
			if len(newpdis_list) != 0:
				ret.append([p_hpo_list, newpdis_list])
		if len(ret) == 0:
			print('warning: dataset is empty!')
		return ret


	def get_dataset(self, data_name, part, min_hpo=0, filter=True):
		"""
		Args:
			part (str): 'validation' | 'test'
		Returns:
			list: [[hpo_list, dis_list], ...]
		"""

		def get_dataset_inner(data_name, part):
			if part == TEST_DATA:
				name_to_path = self.test_to_path
			elif part == VALIDATION_DATA:
				name_to_path = self.valid_to_path
			else:
				assert False
			if data_name not in name_to_path:
				print('Loaded Empty Dataset: {}, {}'.format(data_name, part))
				return []

			dataset = self.get_dataset_with_path(name_to_path[data_name])




			if self.use_rd_code:

				dataset = [[pa[0], source_codes_to_rd_codes(pa[1], self.rd_reader)] for pa in dataset]

			if filter:
				dataset = [patient for patient in dataset if len(patient[0]) >= min_hpo] if min_hpo != 0 else dataset
				dataset = self.dis_filter(dataset)


			return dataset
		if part == VALIDATION_TEST_DATA:
			return get_dataset_inner(data_name, VALIDATION_DATA) + get_dataset_inner(data_name, TEST_DATA)



		return get_dataset_inner(data_name, part)


	def get_utrain_dataset(self, data_name, min_hpo=0):
		"""
			list: [hpo_list]
		"""
		udataset = self.get_utrain_data_set_with_path(self.u_train_to_path[data_name])
		udataset = [hpo_list for hpo_list in udataset if len(hpo_list) >= min_hpo] if min_hpo != 0 else udataset
		return udataset


	def get_combined_u_datasets(self, data_names, min_hpo=0, del_dup=False):
		dataset = []
		for dname in data_names:
			dataset.extend(self.get_utrain_dataset(dname, min_hpo))
		if del_dup:
			dataset = del_obj_list_dup(dataset, lambda hpo_list: tuple(sorted(hpo_list)))
		return dataset


	def get_train_dataset(self):
		"""
			list: [hpo_list, dis_list]
		"""
		return [[hpo_list, [dis_code]] for dis_code, hpo_list in self.get_hpo_reader().get_dis_to_hpo_dict(PHELIST_REDUCE).items()]


	def get_utrain_data_set_with_path(self, path):
		return json.load(open(path))


	def get_dataset_with_path(self, path):

		dataset = json.load(open(path))

		self.standardize_dataset(dataset)
		return dataset


	def standardize_dataset(self, dataset):
		for i in range(len(dataset)):
			if isinstance(dataset[i][1], str):
				dataset[i][1] = [dataset[i][1]]


	def split_dataset(self, dataset):
		"""
		Returns:
			list: validation set
			list: test set
		"""
		return train_test_split(dataset, test_size=self.TEST_SIZE)


	def filter_patients(self, patients):
		pg = PatientGenerator(hpo_reader=self.hpo_reader)
		ret_patients = []
		for hpo_codes, dis_codes in patients:
			assert isinstance(dis_codes, list)
			hpo_codes = pg.process_pa_hpo_list(hpo_codes, reduce=True)
			dis_codes = pg.process_pa_dis_list(dis_codes)
			if len(hpo_codes) < self.VALID_TEST_MIN_HPO_NUM:
				continue
			if len(dis_codes) == 0:
				continue
			ret_patients.append([hpo_codes, dis_codes])
		return ret_patients


	def dataset_statistics(self, dname, keep_general_dis_map=True):
		def rm_general_dis_codes(dis_codes):
			tmp_set = set()
			for dis_code in dis_codes:
				if dis_code in self.keep_dis_to_remove:
					tmp_set.update(self.keep_dis_to_remove[dis_code])
			return [dis_code for dis_code in dis_codes if dis_code not in tmp_set]
		patients = []

		if dname in self.test_names:
			test_patients = self.get_dataset_with_path(self.test_to_path[dname])
			patients.extend(test_patients)
			print(dname, 'test:', len(patients))
		if dname in self.valid_names:
			validation_patients = self.get_dataset_with_path(self.valid_to_path[dname])
			patients.extend(validation_patients)
			print(dname, 'valid:', len(validation_patients))
		if not keep_general_dis_map:
			patients = [[hpo_list, rm_general_dis_codes(dis_list)] for hpo_list, dis_list in patients]
		LabeledDatasetExplainer(patients).explain_save_json(os.path.join(self.TEST_VAL_STATISTIC_FOLDER, f'{dname}.json'))



	def multi_dataset_statistics(self, data_names, keep_general_dis_map=True):
		for dname in data_names:
			self.dataset_statistics(dname, keep_general_dis_map=keep_general_dis_map)


	def gen_test_valid_split_dataset(self):
		save_func = get_save_func(JSON_FILE_FORMAT)

		for dname in ['HMS']:
			dataset = self.get_dataset_with_path(self.origin_to_path[dname])
			dataset = self.filter_patients(dataset)
			valid_set, test_set = self.split_dataset(dataset)
			save_func(valid_set, self.valid_to_path[dname])
			save_func(test_set, self.test_to_path[dname])


	def gen_test_dataset(self):
		save_func = get_save_func(JSON_FILE_FORMAT)

		for dname in ['SIM_NOISE_5.0', 'SIM_NOISE_10.0', 'SIM_NOISE_20.0', 'SIM_NOISE_40.0']:
			if dname in self.valid_names:
				continue
			dataset = self.get_dataset_with_path(self.origin_to_path[dname])
			dataset = self.filter_patients(dataset)
			save_func(dataset, self.test_to_path[dname])


	@check_return('hpo_reader')
	def get_hpo_reader(self):
		return HPOReader()


	@check_return('rd_reader')
	def get_rd_reader(self):
		return RDReader()


	@check_return('all_dis_set')
	def get_all_dis_set(self):
		return set(self.get_hpo_reader().get_dis_list())


	def get_train_XY(self, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1,
				x_sparse=True, xdtype=np.int32, y_one_hot=True, ydtype=np.int32):
		"""
		Args:
			data_name (str)
			part (str): 'train' | 'validation' | 'test' | 'unlabeled'
		Returns:
			csr_matrix or np.ndarray: X; shape=(data_size, hpo_num)
			csr_matrix or np.ndarray: y_; shape(csr)=(data_size, dis_num); shape(ndarray)=(data_size,)
		"""
		return self.get_train_X(phe_list_mode, vec_type, x_sparse, dtype=xdtype), self.get_train_y(y_one_hot, ydtype)


	def getCombinedUTrainXY(self, data_names, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1,
			x_sparse=True, xdtype=np.int32, ydtype=np.int32, min_hpo=0, del_dup=False):
		"""
		Returns:
			csr_matrix or np.ndarray: X; shape=(data_size, hpo_num)
			np.ndarray: y_; shape(ndarray)=(data_size,)
		"""
		udataset = self.get_combined_u_datasets(data_names, min_hpo, del_dup)
		X = self.hpo_lists_to_X(udataset, phe_list_mode, vec_type, x_sparse, xdtype)
		y_ = self.label_lists_to_matrix([[-1]] * X.shape[0], col_num=self.get_hpo_reader().get_dis_num(), one_hot=False, dtype=ydtype)
		return X, y_


	def getUTrainXY(self, data_name, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1,
			x_sparse=True, xdtype=np.int32, ydtype=np.int32, min_hpo=0):
		"""
		Returns:
			csr_matrix or np.ndarray: X; shape=(data_size, hpo_num)
			np.ndarray: y_; shape(ndarray)=(data_size,)
		"""
		udataset = self.get_utrain_dataset(data_name, min_hpo)
		X = self.hpo_lists_to_X(udataset, phe_list_mode, vec_type, x_sparse, xdtype)
		y_ = self.label_lists_to_matrix([[-1]] * X.shape[0], col_num=self.get_hpo_reader().get_dis_num(), one_hot=False, dtype=ydtype)
		return X, y_


	def get_test_val_Xy(self, data_name, part=TEST_DATA, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1,
			x_sparse=True, xdtype=np.int32, y_one_hot=True, ydtype=np.int32, min_hpo=0, use_rd_mix_code=False):
		"""
		Args:
			data_name (str)
			part (str): 'validation' | 'test'
		Returns:
			csr_matrix or np.ndarray: X; shape=(data_size, hpo_num)
			(csr_matrix or np.ndarray, list): (y_, y_column_names); shape(y_)=(data_size, dis_num); shape(ndarray)=(data_size,)
		"""
		dataset = self.get_dataset(data_name, part, min_hpo)
		hpo_lists, dis_lists = zip(*dataset)
		X = self.hpo_lists_to_X(hpo_lists, phe_list_mode, vec_type, x_sparse, xdtype, auto_drop=True)
		y_and_col_names = self.dis_lists_to_y(dis_lists, y_one_hot, ydtype, use_rd_mix_code=use_rd_mix_code)
		return X, y_and_col_names


	def get_levels_test_val_Xy(self, data_name, part=TEST_DATA, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1,
			x_sparse=True, xdtype=np.int32, ydtype=np.int32, min_hpo=0, level_orders_list=None, use_rd_mix_code=False,
			restrict=False, ances_dp=1.0):
		"""
		Returns:
			csr_matrix: X
			(list of csr_matrix, list of list): [level_y_mat, ...], [codes, ...]
		"""
		dataset = self.get_dataset(data_name, part, min_hpo)
		hpo_lists, dis_lists = zip(*dataset)
		X = self.hpo_lists_to_X(hpo_lists, phe_list_mode, vec_type, x_sparse, xdtype, auto_drop=True)

		keep_dis_codes = self.hpo_reader.get_dis_list()
		y_ = self.dis_lists_to_many_level_y(dis_lists, keep_dis_codes, level_orders_list, ydtype, use_rd_mix_code, restrict, ances_dp=ances_dp)
		return X, y_


	def dis_lists_to_y(self, dis_lists, one_hot=True, dtype=np.int32, use_rd_mix_code=False):
		"""
		Args:
			dis_lists (list): [[dis_code, ...], ...]
			list: code list for columns of y_
		"""
		if use_rd_mix_code:
			rd_reader = RDFilterReader(keep_source_codes=self.hpo_reader.get_dis_list())
			rd_dis_num = rd_reader.get_rd_num()
			source_to_rd, rd_map_rank, ret_rd_list = rd_reader.get_source_to_rd(), rd_reader.get_rd_map_rank(), rd_reader.get_rd_list()
			return self.label_lists_to_matrix(
				[ [rd_map_rank[source_to_rd[source_code]] for source_code in source_codes] for source_codes in dis_lists],
				rd_dis_num, one_hot, dtype), ret_rd_list
		else:
			hpo_reader = self.get_hpo_reader()
			dis_int_lists = [item_list_to_rank_list(dis_list, hpo_reader.get_dis_map_rank()) for dis_list in dis_lists]
			return self.label_lists_to_matrix(dis_int_lists, hpo_reader.get_dis_num(), one_hot, dtype), hpo_reader.get_dis_list()


	def hpo_lists_to_X(self, hpo_lists, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, sparse=True, dtype=np.int32, auto_drop=False):
		"""
		Args:
			hpo_lists (list): [[hpo_code1, ...], ...]
		"""
		hpo_int_lists = [item_list_to_rank_list(hpo_list, self.get_hpo_reader().get_hpo_map_rank(), auto_drop) for hpo_list in hpo_lists]
		return self.hpo_int_lists_to_X(hpo_int_lists, phe_list_mode, vec_type, sparse, dtype)


	def _hpo_int_lists_to_raw_X_with_ances_dict_multi(self, hpo_int_lists, ances_dict, phe_list_mode, cpu_use, chunk_size):
		raw_X_list = []
		with Pool(cpu_use) as pool:
			sample_size = len(hpo_int_lists)
			if chunk_size is None:
				chunk_size = max(min(sample_size // cpu_use, 20000), 5000)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			para_list = [(hpo_int_lists[intervals[i]: intervals[i + 1]], ances_dict, phe_list_mode) for i in range(len(intervals) - 1)]
			for raw_X in tqdm(pool.imap(self._hpo_int_lists_to_raw_X_with_ances_dict_multi_wrap, para_list), total=len(para_list), leave=False):
				raw_X_list.extend(raw_X)
		return raw_X_list


	def _hpo_int_lists_to_raw_X_with_ances_dict_multi_wrap(self, para):
		return self._hpo_int_lists_to_raw_X_with_ances_dict(*para)


	def _hpo_int_lists_to_raw_X_with_ances_dict(self, hpo_int_lists, ances_dict, phe_list_mode):
		if phe_list_mode == PHELIST_ANCESTOR:
			return [get_all_ancestors_for_many_with_ances_dict(hpo_int_list, ances_dict) for hpo_int_list in tqdm(hpo_int_lists)]
		elif phe_list_mode == PHELIST_REDUCE:
			return [delete_redundacy_with_ances_dict(hpo_int_list, ances_dict) for hpo_int_list in tqdm(hpo_int_lists)]
		elif phe_list_mode == PHELIST_ANCESTOR_DUP:
			return [get_all_dup_ancestors_for_many_with_ances_dict(hpo_int_list, ances_dict) for hpo_int_list in tqdm(hpo_int_lists)]
		assert False


	def hpo_int_lists_to_raw_X_with_ances_dict(self, hpo_int_lists, ances_dict, phe_list_mode=PHELIST_ANCESTOR, cpu_use=1, chunk_size=None):
		if cpu_use == 1:
			return self._hpo_int_lists_to_raw_X_with_ances_dict(hpo_int_lists, ances_dict, phe_list_mode)
		return self._hpo_int_lists_to_raw_X_with_ances_dict_multi(hpo_int_lists, ances_dict, phe_list_mode, cpu_use, chunk_size)


	def _hpo_int_lists_to_raw_X_multi(self, hpo_int_lists, phe_list_mode, cpu_use, chunk_size):
		raw_X_list = []
		with Pool(cpu_use) as pool:
			sample_size = len(hpo_int_lists)
			if chunk_size is None:
				chunk_size = max(min(sample_size // cpu_use, 20000), 5000)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			para_list = [(hpo_int_lists[intervals[i]: intervals[i + 1]], phe_list_mode) for i in range(len(intervals) - 1)]
			for X in tqdm(pool.imap(self._hpo_int_lists_to_raw_X_multi_wrap, para_list), total=len(para_list), leave=False):
				raw_X_list.extend(X)
		return raw_X_list


	def _hpo_int_lists_to_raw_X_multi_wrap(self, para):
		return self._hpo_int_lists_to_raw_X(*para)


	def _hpo_int_lists_to_raw_X(self, hpo_int_lists, phe_list_mode):
		"""
		Returns:
			list: [[hpo_int1, hpo_int2, ...], ...]; len() = dis_num
		"""
		hpo_int_dict = self.get_hpo_reader().get_hpo_int_dict()
		if phe_list_mode == PHELIST_ANCESTOR:
			return [list(get_all_ancestors_for_many(hpo_int_list, hpo_int_dict)) for hpo_int_list in tqdm(hpo_int_lists)]
		elif phe_list_mode == PHELIST_REDUCE:
			return [delete_redundacy(hpo_int_list, hpo_int_dict) for hpo_int_list in tqdm(hpo_int_lists)]
		elif PHELIST_ANCESTOR_DUP:
			return [get_all_dup_ancestors_for_many(hpo_int_list, hpo_int_dict) for hpo_int_list in tqdm(hpo_int_lists)]
		assert False


	def hpo_int_lists_to_raw_X(self, hpo_int_lists, phe_list_mode=PHELIST_ANCESTOR, cpu_use=1, chunk_size=None):
		if cpu_use == 1:
			return self._hpo_int_lists_to_raw_X(hpo_int_lists, phe_list_mode)
		return self._hpo_int_lists_to_raw_X_multi(hpo_int_lists, phe_list_mode, cpu_use, chunk_size)


	def _hpo_int_lists_to_X_with_ances_dictMulti(self, hpo_int_lists, ances_dict, phe_list_mode,
									vec_type, sparse, dtype, preprocess, cpu_use=12, chunk_size=None):
		XList = []
		with Pool(cpu_use) as pool:
			sample_size = len(hpo_int_lists)
			if chunk_size is None:
				chunk_size = max(min(sample_size // cpu_use, 20000), 5000)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			para_list = [(hpo_int_lists[intervals[i]: intervals[i + 1]], ances_dict, phe_list_mode, vec_type, sparse, dtype, preprocess)
						for i in range(len(intervals) - 1)]
			for X in tqdm(pool.imap(self.hpo_int_lists_to_X_with_ances_dict_multi_wrap, para_list), total=len(para_list), leave=False):
				XList.append(X)
		if sparse == True:
			return vstack(XList, 'csr')
		return np.vstack(XList)


	def hpo_int_lists_to_X_with_ances_dict_multi_wrap(self, para):
		return self._hpo_int_lists_to_X_with_ances_dict(*para)


	def _hpo_int_lists_to_X_with_ances_dict(self, hpo_int_lists, ances_dict, phe_list_mode, vec_type, sparse, dtype, preprocess):
		if preprocess:
			hpo_int_lists = self.hpo_int_lists_to_raw_X_with_ances_dict(hpo_int_lists, ances_dict, phe_list_mode)
		hpo_num = self.get_hpo_reader().get_hpo_num()
		return self.col_lists_to_matrix(hpo_int_lists, hpo_num, dtype, vec_type, sparse)


	def hpo_int_lists_to_X_with_ances_dict(self, hpo_int_lists, ances_dict, phe_list_mode=PHELIST_ANCESTOR,
								vec_type=VEC_TYPE_0_1, sparse=True, dtype=np.int32, preprocess=True, cpu_use=1, chunk_size=None):
		if cpu_use == 1:
			return self._hpo_int_lists_to_X_with_ances_dict(hpo_int_lists, ances_dict, phe_list_mode, vec_type, sparse, dtype, preprocess)
		return self._hpo_int_lists_to_X_with_ances_dictMulti(hpo_int_lists, ances_dict, phe_list_mode, vec_type, sparse, dtype, preprocess, cpu_use, chunk_size)


	def hpo_int_lists_to_X(self, hpo_int_lists, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, sparse=True,
						dtype=np.int32, preprocess=True, cpu_use=1, chunk_size=None):
		"""
		Returns:
			csr_matrix or np.ndarray: shape=[data_size, hpo_num]
		"""
		if preprocess:
			hpo_int_lists = self.hpo_int_lists_to_raw_X(hpo_int_lists, phe_list_mode)
		return self.col_lists_to_matrix(hpo_int_lists, self.get_hpo_reader().get_hpo_num(), dtype, vec_type, sparse, cpu_use, chunk_size)


	def get_train_raw_Xy(self, phe_list_mode, use_rd_mix_code=False, multi_label=False, ret_y_lists=False):
		"""
		Returns:
			list: [[hpo_int1, hpo_int2, ...], ...]; len() = dis_num
			list: [dis_int, dis_int, ...] or [[dis_int], ...]; len() = dis_num
		"""
		if multi_label:
			return self.get_multilabel_train_raw_Xy(phe_list_mode)
		return self.get_single_label_train_raw_Xy(phe_list_mode, use_rd_mix_code, ret_y_lists)


	def get_single_label_train_raw_Xy(self, phe_list_mode, use_rd_mix_code=False, ret_y_lists=False):
		"""
		Returns:
			list: [[hpo_int1, hpo_int2, ...], ...]; len() = dis_num
			list: [dis_int, dis_int, ...] or [[dis_int], ...]; len() = dis_num
		"""
		hpo_reader = self.get_hpo_reader()
		dis_num = hpo_reader.get_dis_num()
		dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(phe_list_mode)
		raw_X = [dis_int_to_hpo_int[i] for i in range(dis_num)]
		if use_rd_mix_code:
			dis_list = self.hpo_reader.get_dis_list()
			rd_reader = RDFilterReader(keep_source_codes=dis_list)
			source_map_rd = rd_reader.get_source_to_rd()
			rd_map_rank = rd_reader.get_rd_map_rank()
			y_ = [ rd_map_rank[source_map_rd[dis_list[i]]] for i in range(dis_num) ]
		else:
			y_ = list(range(len(raw_X)))
		if ret_y_lists:
			y_ = [[dis_int] for dis_int in y_]
		return raw_X, y_


	def get_multilabel_train_raw_Xy(self, phe_list_mode):
		"""
		Returns:
			list: [[hpo_int1, hpo_int2, ...], ...]; len() = dis_num
			list: [[dis_int1, ...], [dis_int2, ...], ...]; len() = dis_num
		"""
		hpo_reader = self.get_hpo_reader()

		dis_map_rank = hpo_reader.get_dis_map_rank()
		dis_list = hpo_reader.get_dis_list(); dis_set = set(dis_list)
		hpo_map_rank = hpo_reader.get_hpo_map_rank()
		dis_to_hpo = hpo_reader.get_dis_to_hpo_dict(phe_list_mode)
		raw_X = [item_list_to_rank_list(dis_to_hpo[dis_code], hpo_map_rank) for dis_code in dis_list]
		rd_reader = RDReader(); source_to_rd = rd_reader.get_source_to_rd(); rd_dict = rd_reader.get_rd_dict()
		y_ = []
		for dis_code in dis_list:
			cand_dis_codes = rd_dict[source_to_rd[dis_code]]['SOURCE_CODES']
			y_.append([dis_map_rank[code] for code in cand_dis_codes if code in dis_set])
		return raw_X, y_


	def get_train_X(self, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, sparse=True, dtype=np.int32, dp=1.0, up_induce_rule='max'):
		"""
		Returns:
			csr_matrix or np.ndarray: shape=[dis_num, hpo_num]
		"""
		if vec_type == VEC_TYPE_PROB or vec_type == VEC_TYPE_LOG_PROB:
			return self.get_train_prob_X(dp, phe_list_mode, vec_type, up_induce_rule, sparse, dtype)
		hpo_reader = self.get_hpo_reader()
		data_size, hpo_num = hpo_reader.get_dis_num(), hpo_reader.get_hpo_num()
		dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(phe_list_mode)
		hpo_int_lists = [dis_int_to_hpo_int[i] for i in range(data_size)]
		return self.col_lists_to_matrix(hpo_int_lists, hpo_num, dtype, vec_type, sparse)


	def rd_y_to_source_y(self, rd_y, rd_list, rd_to_sources, col_source_list):
		"""Note: source codes that are not in source_col_list will be dropped
		Args:
			rd_y (csr_matrix): shape(csr)=(data_size, dis_num); shape(ndarray)=(data_size,)
		Returns:
			csr_matrix: shape=(data_size, class_num)
		"""
		assert isinstance(rd_y, csr_matrix)
		source_map_rank = {source_code: i for i, source_code in enumerate(col_source_list)}
		data_size, feature_size = rd_y.shape[0], len(col_source_list)
		mx = rd_y.tocoo()
		data, rows, cols = mx.data, mx.row, mx.col
		new_data, new_rows, new_cols = [], [], []
		for v, r, c in zip(data, rows, cols):
			cols = [ source_map_rank[source_code] for source_code in rd_to_sources[rd_list[c]] if source_code in source_map_rank]
			new_data.extend([v] * len(cols))
			new_rows.extend([r] * len(cols))
			new_cols.extend(cols)
		return csr_matrix((new_data, (new_rows, new_cols)), shape=(data_size, feature_size), dtype=rd_y.dtype)


	def get_levels_train_y(self, level_orders_list=None, dtype=np.float32, use_rd_mix_code=False, restrict=False, ances_dp=1.0):
		"""
		Returns:
			list: [csr_matrix, ...]
			list: [[code1, code2, ...], ...]
		"""
		keep_dis_codes = self.hpo_reader.get_dis_list()
		dis_lists = [[dis_code] for dis_code in keep_dis_codes]
		return self.dis_lists_to_many_level_y(dis_lists, keep_dis_codes, level_orders_list, dtype, use_rd_mix_code, restrict, ances_dp=ances_dp)


	def dis_lists_to_many_level_y(self, dis_lists, keep_dis_codes, level_orders_list=None, dtype=np.int32, use_rd_mix_code=False,
			restrict=False, ances_dp=1.0):
		"""
		Returns:
			list: [csr_matrix, ...]
			list: [[code1, code2, ...], ...]
		"""
		if level_orders_list is None:
			level_orders_list = [[DISORDER_GROUP_LEVEL], [DISORDER_LEVEL], [DISORDER_SUBTYPE_LEVEL]]
		level_mat_list, level_codes_list = [], []
		for level_orders in level_orders_list:
			level_mat, level_codes = self.dis_lists_to_level_y(dis_lists, keep_dis_codes, level_orders,
				dtype=dtype, use_rd_mix_code=use_rd_mix_code, restrict=restrict, ances_dp=ances_dp)
			level_mat_list.append(level_mat)
			level_codes_list.append(level_codes)
		return level_mat_list, level_codes_list


	def dis_lists_to_level_y(self, dis_lists, keep_dis_codes, level_orders, dtype=np.int32, use_rd_mix_code=False, restrict=False, ances_dp=1.0):
		"""
		Args:
			dis_lists (list): [[dis_code1, ...], ...]
			level_orders (list): e.g. [DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
		Returns:
			csr_matrix: shape=(data_size, class_num)
			list: code list for columns of y_; [code1, code2, ...]
		"""
		y_, col_dis_names, rd_reader = self.dis_lists_to_multi_label_y_expand(dis_lists, keep_dis_codes, dtype=dtype,
			use_rd_mix_code=use_rd_mix_code, ances_expand=True, ret_rd_reader=True, restrict=restrict, ances_dp=ances_dp)
		code_to_level = rd_reader.get_rd_to_level() if use_rd_mix_code else rd_reader.get_source_to_level()
		code_map_old_rank = {code: i for i, code in enumerate(col_dis_names)}

		level_codes_list = [[] for _ in level_orders]
		for code in col_dis_names:
			code_level = code_to_level[code]
			for i, level in enumerate(level_orders):
				if code_level == level:
					level_codes_list[i].append(code)
					break
		reorder_code_list = [code for level_codes in level_codes_list for code in level_codes]
		code_map_new_rank = {code: i for i, code in enumerate(reorder_code_list)}
		old_int_map_new_int = {code_map_old_rank[code]: code_map_new_rank[code] for code in reorder_code_list}
		return self.reorder_sp_col(y_, old_int_map_new_int, len(reorder_code_list)), reorder_code_list


	def reorder_sp_col(self, m, old_col_to_new_col, col_num):
		"""Note: columns not in old_col_to_new_col will be dropped
		Args:
			m (csr_matrix)
			old_col_to_new_col (dict): {old_col: new_col}
		Returns:
			csr_matrix
		"""
		m = m.tocoo()
		data, rows, cols = m.data, m.row, m.col
		new_data, new_rows, new_cols = [], [], []
		for d, r, c in zip(data, rows, cols):
			if c in old_col_to_new_col:
				new_data.append(d); new_rows.append(r); new_cols.append(old_col_to_new_col[c])
		return csr_matrix((new_data, (new_rows, new_cols)), shape=(m.shape[0], col_num), dtype=m.dtype)


	def get_train_y(self, one_hot=True, dtype=np.int32, use_rd_mix_code=False, multi_label=False,
			expand_ances=False, expand_desc=False, desc_dp=None, ances_dp=1.0):
		"""
		Returns:
			csr_matrix or np.ndarray: y_; shape(csr)=(data_size, dis_num); shape(ndarray)=(data_size,)
			list: code list for columns of y_
		"""
		if multi_label:
			assert one_hot
			if expand_ances or expand_desc:
				return self.get_multi_label_train_y_expand(dtype=dtype, use_rd_mix_code=use_rd_mix_code,
					ances_expand=expand_ances, desc_expand=expand_desc, desc_dp=desc_dp, ances_dp=ances_dp)
			else:
				return self.get_multi_label_train_y(dtype=dtype, use_rd_mix_code=use_rd_mix_code)
		return self.get_single_label_train_y(one_hot=one_hot, dtype=dtype, use_rd_mix_code=use_rd_mix_code)


	def get_single_label_train_y(self, one_hot=True, dtype=np.float32, use_rd_mix_code=False):
		"""
		Returns:
			csr_matrix or np.ndarray: y_; shape(csr)=(data_size, dis_num); shape(ndarray)=(data_size,)
			list: code list for columns of y_
		"""
		dis_lists = [[dis_code] for dis_code in self.hpo_reader.get_dis_list()]
		return self.dis_lists_to_y(dis_lists, one_hot=one_hot, dtype=dtype, use_rd_mix_code=use_rd_mix_code)


	def get_multi_label_train_y(self, dtype=np.float32, use_rd_mix_code=False):
		"""
		Returns:
			csr_matrix: shape=(data_size, dis_num)
			list: code list for columns of y_
		"""
		rd_y, col_rd_codes = self.get_single_label_train_y(one_hot=True, dtype=dtype, use_rd_mix_code=True)
		if not use_rd_mix_code:
			col_source_codes = self.hpo_reader.get_dis_list()
			rd_to_sources = RDFilterReader(keep_source_codes=col_source_codes).get_rd_to_sources()
			y = self.rd_y_to_source_y(rd_y, rd_list=col_rd_codes, rd_to_sources=rd_to_sources, col_source_list=col_source_codes)
			return y, col_source_codes
		return rd_y, col_rd_codes


	def dis_lists_to_multi_label_y_expand(self, dis_lists, keep_dis_codes, dtype=np.float32, use_rd_mix_code=False,
			ances_expand=False, desc_expand=False, desc_dp=None, ret_rd_reader=False, restrict=False, ances_dp=1.0):
		"""
		Returns:
			csr_matrix: y_; shape=(len(dis_lists), dis_num)
			list: code list for columns of y_;
		"""
		data_size = len(dis_lists)
		rd_reader = RDFilterReader(keep_source_codes=keep_dis_codes, keep_ances=ances_expand)
		rd_dis_num = rd_reader.get_rd_num()
		rd_dict = rd_reader.get_rd_dict()
		source_to_rd, rd_map_rank = rd_reader.get_source_to_rd(), rd_reader.get_rd_map_rank()
		data, rows, cols = [], [], []
		for i in range(data_size):
			for dis_code in dis_lists[i]:
				rd_code = source_to_rd[dis_code]
				data.append(1.0); rows.append(i); cols.append(rd_map_rank[rd_code])
				if ances_expand:
					ances_rds = get_all_ancestors(rd_code, rd_dict, contain_self=False)
					data.extend([ances_dp] * len(ances_rds))
					rows.extend([i] * len(ances_rds))
					cols.extend([rd_map_rank[ances_rd] for ances_rd in ances_rds])
				if desc_expand:
					desc_rds = get_all_descendents(rd_code, rd_dict, contain_self=False)
					if desc_dp is None:
						desc_dp = (1.0 / len(desc_rds)) if len(desc_rds) > 0 else 0.0
					data.extend([desc_dp] * len(desc_rds))
					rows.extend([i] * len(desc_rds))
					cols.extend([rd_map_rank[desc_rd] for desc_rd in desc_rds])
		rd_y = csr_matrix((data, (rows, cols)), shape=(data_size, rd_dis_num), dtype=dtype)
		if use_rd_mix_code:
			y = rd_y
			ret_code_list = rd_reader.get_rd_list()
		else:
			rd_list, rd_to_sources = rd_reader.get_rd_list(), rd_reader.get_rd_to_sources()
			ret_code_list = rd_reader.get_source_list() if ances_expand or desc_expand else self.hpo_reader.get_dis_list()
			y = self.rd_y_to_source_y(rd_y, rd_list=rd_list, rd_to_sources=rd_to_sources, col_source_list=ret_code_list)
			if restrict:
				source_map_rank = {code:i for i, code in enumerate(ret_code_list)}
				rm_int_lists = []
				for i, dis_list in enumerate(dis_lists):
					dis_ex_set = set([source_code_ex for source_code in dis_list for source_code_ex in rd_to_sources[source_to_rd[source_code]]])
					set_zero_source_codes = dis_ex_set - set(dis_list)
					rm_int_lists.append([source_map_rank[code] for code in set_zero_source_codes])
				y = y - self.label_lists_to_matrix(rm_int_lists, y.shape[1], dtype=y.dtype)
		assert len(unique_list(ret_code_list)) == len(ret_code_list)
		return (y, ret_code_list, rd_reader) if ret_rd_reader else (y, ret_code_list)


	def get_multi_label_train_y_expand(self, dtype=np.float32, use_rd_mix_code=False,
			ances_expand=False, desc_expand=False, desc_dp=None, ances_dp=1.0):
		"""
		Returns:
			csr_matrix: shape=(data_size, dis_num)
			list: code list for columns of y_
		"""
		keep_dis_list = self.hpo_reader.get_dis_list()
		dis_lists = [[dis_code] for dis_code in keep_dis_list]
		return self.dis_lists_to_multi_label_y_expand(dis_lists, keep_dis_list, dtype=dtype, use_rd_mix_code=use_rd_mix_code,
			ances_expand=ances_expand, desc_expand=desc_expand, desc_dp=desc_dp, ret_rd_reader=False, ances_dp=ances_dp)


	def get_train_prob_X(self, dp=1.0, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_PROB,
					up_induce_rule='max', sparse=True, dtype=np.float32):
		"""
		Args:
			dp (float)
			phe_list_mode (str): PHELIST_ANCESTOR | PHELIST_REDUCE
			vec_type (str): VEC_TYPE_PROB | VEC_TYPE_LOG_PROB
			up_induce_rule (str): 'max' | 'sum' | 'ind'
		Returns:
			csr_matrix or np.ndarray: shape=[dis_num, hpo_num]
		"""
		def get_csr_mat(d):
			# d = {dis_int: {hpo_int: prob}}
			data, rows, cols = [], [], []
			for dis_int in range(DIS_NUM):
				hpo_int_to_prob = d[dis_int]
				rcols, rdata = zip(*hpo_int_to_prob.items())
				rows.extend([dis_int] * len(rdata))
				cols.extend(rcols)
				data.extend(rdata)
			for v in data:
				if not (type(v) == float or type(v) == np.float64):
					print(v, type(v))
					assert False
			for v in rows:
				if not type(v) == int:
					print('rows:', v, type(v))
					assert False
			for v in cols:
				if not type(v) == int:
					print('cols:', v, type(v))
					assert False
			return csr_matrix((data, (rows, cols)), shape=(DIS_NUM, HPO_NUM), dtype=dtype)
		assert phe_list_mode == PHELIST_ANCESTOR or phe_list_mode == PHELIST_REDUCE
		assert vec_type == VEC_TYPE_PROB or VEC_TYPE_LOG_PROB
		hpo_reader = self.get_hpo_reader()
		DIS_NUM, HPO_NUM = hpo_reader.get_dis_num(), hpo_reader.get_hpo_num()
		dis_int_to_hpo_int_prob = hpo_reader.get_dis_int_to_hpo_int_prob(default_prob=dp, phe_list_mode=PHELIST_REDUCE)

		for dis_int, hpo_int_prob_list in dis_int_to_hpo_int_prob.items():
			for hpo_int, prob in hpo_int_prob_list:
				assert prob is not None

		if phe_list_mode == PHELIST_REDUCE:
			d = {dis_int: {hpo_int: prob for hpo_int, prob in hpo_int_probs} for dis_int, hpo_int_probs in dis_int_to_hpo_int_prob.items()}
		else:
			d = {}
			ROOT_HPO_INT = hpo_reader.get_hpo_map_rank()[ROOT_HPO_CODE]
			hpo_int_dict = hpo_reader.get_hpo_int_dict()
			dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR)
			for dis_int in range(DIS_NUM):
				hpo_int_to_prob = {hpo_int:None for hpo_int in dis_int_to_hpo_int[dis_int]}
				hpo_int_to_prob.update({hpo_int:prob for hpo_int, prob in dis_int_to_hpo_int_prob[dis_int]})
				self.get_prob_tree(ROOT_HPO_INT, hpo_int_to_prob, hpo_int_dict, up_induce_rule)
				for hpo_int in hpo_int_to_prob.keys():
					if hpo_int_to_prob[hpo_int] is None:
						self.get_prob_tree(hpo_int, hpo_int_to_prob, hpo_int_dict, up_induce_rule)
				d[dis_int] = hpo_int_to_prob

		m = get_csr_mat(d)
		assert np.logical_and(m.data >= 0.0, m.data <= 1.0).all()
		if vec_type == VEC_TYPE_LOG_PROB:
			m.data = np.log(m.data)
		if not sparse:
			m = m.toarray()
		return m


	def get_prob_tree(self, root_int, hpo_int_to_prob, hpo_int_dict, up_induce_rule):
		if hpo_int_to_prob[root_int] is not None:
			return hpo_int_to_prob[root_int]
		child_prob_list = [self.get_prob_tree(child_int, hpo_int_to_prob, hpo_int_dict, up_induce_rule)
			for child_int in hpo_int_dict[root_int]['CHILD'] if child_int in hpo_int_to_prob]
		root_prob = self.cal_parent_prob(child_prob_list, up_induce_rule)
		hpo_int_to_prob[root_int] = root_prob
		return root_prob


	def cal_parent_prob(self, child_prob_list, up_induce_rule):
		if up_induce_rule == 'sum':
			return min(sum(child_prob_list), 1.0)
		elif up_induce_rule == 'max':
			return max(child_prob_list)
		elif up_induce_rule == 'ind':
			child_prob_ary = np.array(child_prob_list)
			return 1 - np.exp(np.log(1 - child_prob_ary).sum())
		else:
			assert False


	def label_lists_to_matrix(self, label_lists, col_num=None, one_hot=True, dtype=np.int32):
		"""
		Args:
			label_lists (list): [int_list, int_list, ...]
		Returns:
			csr_matrix or np.ndarray: y_; shape(csr)=(data_size, dis_num); shape(ndarray)=(data_size,)
		"""
		if one_hot:
			return data_to_01_matrix(label_lists, col_num, dtype=np.int32)
		else:
			for labels in label_lists: assert len(labels) == 1
			return np.array(label_lists, dtype=dtype).flatten()


	def _col_lists_to_matrix_multi(self, data, col_num, dtype, vec_type, sparse, cpu_use=12, chunk_size=None):
		XList = []
		with Pool(cpu_use) as pool:
			sample_size = len(data)
			if chunk_size is None:
				chunk_size = max(min(sample_size//cpu_use, 20000), 5000)
			intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
			para_list = [(data[intervals[i]: intervals[i+1]], col_num, dtype, vec_type, sparse) for i in range(len(intervals) - 1)]
			for X in tqdm(pool.imap(self.col_lists_to_matrix_multi_wrap, para_list), total=len(para_list), leave=False):
				XList.append(X)
		if sparse == True:
			return vstack(XList, 'csr')
		return np.vstack(XList)


	def col_lists_to_matrix_multi_wrap(self, para):
		data, col_num, dtype, vec_type, sparse = para
		return self._col_lists_to_matrix(data, col_num, dtype, vec_type, sparse)


	def _col_lists_to_matrix(self, data, col_num, dtype, vec_type, sparse):

		if vec_type == VEC_TYPE_0_1:
			if sparse:
				return data_to_01_matrix(data, col_num, dtype)
			else:
				return data_to_01_dense_matrix(data, col_num, dtype)
		elif vec_type == VEC_TYPE_TF:
			if sparse:
				return data_to_tf_matrix(data, col_num, dtype)
			else:
				return data_to_tf_dense_matrix(data, col_num, dtype)
		elif vec_type == VEC_TYPE_TF_IDF:
			if sparse:
				return self.data_to_tf_idf_matrix(data, col_num, dtype)
			else:
				return self.data_to_tf_idf_dense_matrix(data, col_num, dtype)
		elif vec_type == VEC_TYPE_IDF:
			if sparse:
				return self.data_to_idf_matrix(data, col_num, dtype)
			else:
				return self.data_to_idf_dense_matrix(data, col_num, dtype)
		assert False


	def col_lists_to_matrix(self, data, col_num, dtype, vec_type, sparse, cpu_use=1, chunk_size=None):
		if cpu_use == 1:
			return self._col_lists_to_matrix(data, col_num, dtype, vec_type, sparse)
		return self._col_lists_to_matrix_multi(data, col_num, dtype, vec_type, sparse, cpu_use, chunk_size)


	def get_tf_idf_transformer(self):
		if self.tf_idf_transformer is None:
			from core.predict.calculator.tfidf_calculator import TFIDFCalculator
			self.tf_idf_transformer = TFIDFCalculator().get_default_transformer()
		return self.tf_idf_transformer


	def data_to_tf_idf_matrix(self, data, col_num, dtype=np.float64):

		transformer = self.get_tf_idf_transformer()
		return transformer.transform(data_to_tf_matrix(data, col_num, dtype=np.int32)).astype(dtype)


	def data_to_tf_idf_dense_matrix(self, data, col_num, dtype=np.float64):

		return self.data_to_tf_idf_matrix(data, col_num, dtype).A


	def data_to_idf_matrix(self, data, col_num, dtype=np.float64):
		transformer = self.get_tf_idf_transformer()
		return transformer.transform(data_to_01_matrix(data, col_num, dtype=np.int32)).astype(dtype)


	def data_to_idf_dense_matrix(self, data, col_num, dtype=np.float64):
		return self.data_to_idf_matrix(data, col_num, dtype).A


	def gen_phenomizer_sample_patients(self, input_folder, data_names):
		from core.utils.utils import get_file_list
		def get_sample_patients_from_folder(folder):
			phe_txts = sorted(get_file_list(folder, lambda p: p.endswith('phe.txt')))
			all_ranks = sorted([int(os.path.split(phe_txt)[1].split('-')[0]) for phe_txt in phe_txts])
			patients = []
			for i in all_ranks:
				hpo_list = open(os.path.join(folder, f'{i}-phe.txt')).read().strip().splitlines()
				dis_list = open(os.path.join(folder, f'{i}-dis.txt')).read().strip().splitlines()
				patients.append([hpo_list, dis_list])
			return patients

		for data_name in data_names:
			sample_patients = get_sample_patients_from_folder(os.path.join(input_folder, data_name))
			json.dump(sample_patients, open(self.test_to_path[data_name], 'w'), indent=2)
			self.dataset_statistics(data_name, keep_general_dis_map=False)


	def gen_phenomizer_sample_more_patients(self, num, data_name, phe_dis_save_folder, gen_before_folders):
		from core.utils.utils import get_file_list
		import random
		def get_ranks_before(gen_before_folders, patients):
			before_ranks = []
			for folder in gen_before_folders:
				phe_txts = sorted(get_file_list(folder, lambda p:p.endswith('phe.txt')))
				before_ranks.extend([int(os.path.split(phe_txt)[1].split('-')[0]) for phe_txt in phe_txts])
				for r in before_ranks:
					assert open(os.path.join(folder, f'{r}-phe.txt')).read().strip().splitlines() == patients[r][0] # check
					assert open(os.path.join(folder, f'{r}-dis.txt')).read().strip().splitlines() == patients[r][1] # check
			before_ranks_set = set(before_ranks)
			assert len(before_ranks_set) == len(before_ranks)
			return before_ranks_set
		patients = self.get_dataset_with_path(self.test_to_path[data_name])
		before_ranks_set = get_ranks_before(gen_before_folders, patients)
		left_ranks = [i for i in range(len(patients)) if i not in before_ranks_set]
		print(data_name, 'Length of before_ranks = {}; Length of left_ranks = {}'.format(len(before_ranks_set), len(left_ranks)))
		if num == 'all' or num > len(left_ranks):
			select_ranks = left_ranks
		else:
			select_ranks = random.sample(left_ranks, num)
		print(data_name, 'Length of select_ranks: {}'.format(len(select_ranks)))
		os.makedirs(phe_dis_save_folder, exist_ok=True)
		for r in sorted(select_ranks):
			open(os.path.join(phe_dis_save_folder, f'{r}-phe.txt'), 'w').writelines('\n'.join(patients[r][0]))
			open(os.path.join(phe_dis_save_folder, f'{r}-dis.txt'), 'w').writelines('\n'.join(patients[r][1]))
		return select_ranks


def check():
	import random
	from core.explainer import Explainer
	explainer = Explainer()

	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])
	dh = DataHelper(hpo_reader=hpo_reader)

	level_y_list, level_codes_list = dh.get_levels_train_y(
		level_orders_list=[[DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]],
		dtype=np.float32, use_rd_mix_code=False, restrict=False)
	for level_y, level_codes in zip(level_y_list, level_codes_list):
		print(level_y.shape, len(level_codes))
		sample_row_indices = random.sample(list(range(level_y.shape[0])), 20)
		for row_idx in sample_row_indices:
			print(f'row = {row_idx} ================')
			cols = level_y[row_idx].indices
			print('\n'.join(explainer.add_cns_info([level_codes[col] for col in cols])))
	pass


if __name__ == '__main__':



	dis_list = "/home/xhmao19/project/hy_works/2020_10_20_RareDisease-master/core/data/preprocess/knowledge/HPO/dis_list.json"

	import json

	with open(dis_list, 'r', encoding='utf8')as fp:
		json_data = json.load(fp)


	rd_reader = RDReader()
	dis_KB_to_RD_num = {}
	result = []
	for i_disease in json_data:
		print('i_disease:%r, and the result is:%r'%(i_disease,source_codes_to_rd_codes([i_disease],rd_reader)))
		result.append(source_codes_to_rd_codes([i_disease],rd_reader)[0])
		dis_KB_to_RD_num[i_disease] = source_codes_to_rd_codes([i_disease],rd_reader)[0]


	saved_json = json.dumps(dis_KB_to_RD_num, indent=2, ensure_ascii=False)
	import codecs
	with codecs.open('/home/xhmao19/project/hy_works/2020_10_20_RareDisease-master/core/data/preprocess/knowledge/HPO/dis_KB_to_RD_num.json', 'a', 'utf-8') as file:
		file.write(saved_json)





	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])

	dh = DataHelper(hpo_reader=hpo_reader)

	dh.multi_dataset_statistics([
		# 'SIM_ORIGIN', 'SIM_NOISE', 'SIM_IMPRE', 'SIM_IMPRE_NOISE', 'SIM_NOISE_IMPRE',
		'HMS', # 'THAN', 'RAMEDIS', 'CJFH', 'PUMC', 'MME',
		# 'RAMEDIS_SAMPLE_100', 'CJFH_SAMPLE_100', 'PUMC_SAMPLE_100', 'MME_SAMPLE_100',
		# 'RAMEDIS_COMP_LB', 'CJFH_COMP_LB', 'PUMC_COMP_LB',
		# 'PUMC_2000_CHPO_E',
	], keep_general_dis_map=True)

