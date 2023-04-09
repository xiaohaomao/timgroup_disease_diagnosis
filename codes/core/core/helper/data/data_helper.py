import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import itertools

from core.utils.constant import DATA_PATH, PHELIST_ANCESTOR, PHELIST_REDUCE, PHELIST_ANCESTOR_DUP, JSON_FILE_FORMAT, VEC_TYPE_IDF, ROOT_HPO_CODE
from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL, SEED, VALIDATION_TEST_DATA
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF, VEC_TYPE_TF_IDF, TEST_DATA, VALIDATION_DATA, VEC_TYPE_PROB, VEC_TYPE_LOG_PROB
from core.utils.utils import item_list_to_rank_list, get_save_func, data_to_01_matrix, data_to_tf_dense_matrix, data_to_tf_matrix, data_to_01_dense_matrix
from core.utils.utils import get_all_ancestors_for_many, delete_redundacy, get_all_dup_ancestors_for_many, del_obj_list_dup, count_obj_list_dup
from core.utils.utils import get_all_ancestors_for_many_with_ances_dict, delete_redundacy_with_ances_dict, get_all_dup_ancestors_for_many_with_ances_dict
from core.utils.utils import slice_list_with_keep_set, check_return, split_path, get_all_ancestors, get_all_descendents, unique_list
from core.utils.utils import combine_key_to_list
from core.reader import HPOReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader, RDReader, RDFilterReader, source_codes_to_rd_codes
from core.helper.data.base_data_helper import BaseDataHelper
from core.explainer.dataset_explainer import LabeledDatasetExplainer
from core.patient import HmsPatientGenerator


class DataHelper(BaseDataHelper):
	def __init__(self, hpo_reader=None, rd_reader=None):
		super(DataHelper, self).__init__()
		integrate_prefix = 'INTEGRATE_'
		self.hpo_reader = hpo_reader or HPOReader()
		self.rd_reader = rd_reader or RDReader()

		self.use_rd_code = self.hpo_reader.name.startswith(integrate_prefix)
		self.all_dis_set = None
		self.tf_idf_transformer = None

		self.TEST_VAL_STATISTIC_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'patient', self.hpo_reader.name, 'test_val_statistics')
		os.makedirs(self.TEST_VAL_STATISTIC_FOLDER, exist_ok=True)

		self.TEST_SIZE = 0.6
		self.VALID_TEST_MIN_HPO_NUM = 3
		self.default_sources = None
		self.keep_dis_to_remove = {
			## RAMEDIS ##
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
			## MME ##
			'OMIM:604317': ['ORPHA:2512'],
			'OMIM:612541': ['CCRD:104'],
		}
		self.keep_dis_to_remove = combine_key_to_list(self.keep_dis_to_remove, HmsPatientGenerator().get_keep_to_general_dis())
		self.dataset_mark = self.hpo_reader.name[len(integrate_prefix):] if self.use_rd_code else self.hpo_reader.name


		self.test_names = [
			## validation subset of RAMEDIS ##
			'Validation_subsets_of_RAMEDIS',

			## Multi-country-test set ##
			'Multi-country-test',

			## combined multi-country set ##
			'Combined-Multi-Country',

			## PUMCH-L datasest ##
			'PUMCH-L-CText2Hpo',
			'PUMCH-L-Meta',
			'PUMCH-L-CHPO',

			## PUMCH-MDT dataset ##
			'PUMCH-MDT',

			## PUMCH-ADM dataset ##
			'PUMCH-ADM',

			## Sampled_100 cases ##
			'Multi-country-test-set-100',
			'RAMEDIS_100',

			## 24 methylmalonic academia cases  using different knowledge bases ##
			'MUT_24_CASES_OMIM',
			'MUT_24_CASES_ORPHA',
			'MUT_24_CASES_CCRD',
			'MUT_24_CASES_OMIM_ORPHA',
			'MUT_24_CASES_CCRD_ORPHA',
			'MUT_24_CASES_CCRD_OMIM',
			'MUT_24_CASES_CCRD_OMIM_ORPHA',

			## validation subsets of RAMEDIS using different knowledge bases ##
			'validation_subset_RAMDEIS_CCRD',
			'validation_subset_RAMDEIS_OMIM',
			'validation_subset_RAMDEIS_ORPHA',
			'validation_subset_RAMDEIS_CCRD_OMIM',
			'validation_subset_RAMDEIS_CCRD_ORPHA',
			'validation_subset_RAMDEIS_OMIM_ORPHA',
			'validation_subset_RAMDEIS_CCRD_OMIM_ORPHA',

			## multi_country_test using different knowledge bases ##
			'Multi-country-test_CCRD',
			'Multi-country-test_OMIM',
			'Multi-country-test_ORPHA',
			'Multi-country-test_CCRD_OMIM',
			'Multi-country-test_CCRD_ORPHA',
			'Multi-country-test_OMIM_ORPHA',
			'Multi-country-test_CCRD_OMIM_ORPHA',

			## simulated datasets ##
			'SIM_ORIGIN',
			'SIM_NOISE',
			'SIM_IMPRE',
			'SIM_IMPRE_NOISE',

		]

		self.test_to_path = {
			## Validation subset of RAMEDIS ##
			'Validation_subsets_of_RAMEDIS': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"Validation_subsets_of_RAMEDIS.json"),
			## Multi-country set dataset ##
			'Multi-country-test': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test', "Multi-country-test.json"),
			## Combined Multi-Country dataset ##
			'Combined-Multi-Country': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"Combined-Multi-Country.json"),
			## PUMCH-L dataset ##
			'PUMCH-L-CText2Hpo':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L.json"),

			## PUMCH-MDT dataset ##
			'PUMCH-MDT': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"PUMCH-MDT.json"),
			## PUMCH-ADM dataset ##
			'PUMCH-ADM': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark,'test',"PUMCH-ADM.json"),
			## PUMCH-L dataset based on Meta Thesaurus phenotype extraction method ##
			'PUMCH-L-Meta': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L-Meta.json"),
			## PUMCH-L dataset based on CHPO phenotype extraction method ##
			'PUMCH-L-CHPO': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', "PUMCH-L-CHPO.json"),
			## sampled 100 cases ##
			'Multi-country-test-set-100':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test-set-SAMPLE_100.json'),
			'RAMEDIS_100':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'RAMEDIS_SAMPLE_100.json'),

			## 24 methylmalonic academia cases  using different knowledge bases ##
			'MUT_24_CASES_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_OMIM.json'),
			'MUT_24_CASES_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_ORPHA.json'),
			'MUT_24_CASES_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD.json'),
			'MUT_24_CASES_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_OMIM_ORPHA.json'),
			'MUT_24_CASES_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_ORPHA.json'),
			'MUT_24_CASES_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_OMIM.json'),
			'MUT_24_CASES_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', '24_methylmalonic_academia_cases_CCRD_OMIM_ORPHA.json'),

			## validation subsets of RAMEDIS using different knowledge bases ##
			'Multi-country-tuning_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD.json'),
			'Multi-country-tuning_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_OMIM.json'),
			'Multi-country-tuning_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_ORPHA.json'),
			'Multi-country-tuning_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_OMIM.json'),
			'Multi-country-tuning_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_ORPHA.json'),
			'Multi-country-tuning_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_OMIM_ORPHA.json'),
			'Multi-country-tuning_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Validation_subsets_of_RAMEDIS_CCRD_OMIM_ORPHA.json'),

			## multi_country_test  using different knowledge bases ##
			'Multi-country-test_CCRD':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD.json'),
			'Multi-country-test_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_OMIM.json'),
			'Multi-country-test_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_ORPHA.json'),
			'Multi-country-test_CCRD_OMIM':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_OMIM.json'),
			'Multi-country-test_CCRD_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_ORPHA.json'),
			'Multi-country-test_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_OMIM_ORPHA.json'),
			'Multi-country-test_CCRD_OMIM_ORPHA':os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'Multi-country-test_CCRD_OMIM_ORPHA.json'),

			## simulated datasets ##
			'SIM_ORIGIN': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_ORIGIN.json'),
			'SIM_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_NOISE.json'),
			'SIM_IMPRE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_IMPRE.json'),
			'SIM_IMPRE_NOISE': os.path.join(DATA_PATH, 'preprocess', 'patient', self.dataset_mark, 'test', 'SIM_IMPRE_NOISE.json'),


		}


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


	def get_dataset_with_path(self, path):

		dataset = json.load(open(path))

		self.standardize_dataset(dataset)
		return dataset


	def standardize_dataset(self, dataset):
		for i in range(len(dataset)):
			if isinstance(dataset[i][1], str):
				dataset[i][1] = [dataset[i][1]]



	@check_return('hpo_reader')
	def get_hpo_reader(self):
		return HPOReader()


	@check_return('rd_reader')
	def get_rd_reader(self):
		return RDReader()


	@check_return('all_dis_set')
	def get_all_dis_set(self):
		return set(self.get_hpo_reader().get_dis_list())



	def _hpo_int_lists_to_X_with_ances_dict(self, hpo_int_lists, ances_dict, phe_list_mode, vec_type, sparse, dtype, preprocess):
		if preprocess:
			hpo_int_lists = self.hpo_int_lists_to_raw_X_with_ances_dict(hpo_int_lists, ances_dict, phe_list_mode)
		hpo_num = self.get_hpo_reader().get_hpo_num()
		return self.col_lists_to_matrix(hpo_int_lists, hpo_num, dtype, vec_type, sparse)


	def hpo_int_lists_to_X(self, hpo_int_lists, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, sparse=True,
						dtype=np.int32, preprocess=True, cpu_use=1, chunk_size=None):
		"""
		Returns:
			csr_matrix or np.ndarray: shape=[data_size, hpo_num]
		"""
		if preprocess:
			hpo_int_lists = self.hpo_int_lists_to_raw_X(hpo_int_lists, phe_list_mode)
		return self.col_lists_to_matrix(hpo_int_lists, self.get_hpo_reader().get_hpo_num(), dtype, vec_type, sparse, cpu_use, chunk_size)

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



if __name__ == '__main__':
	pass




