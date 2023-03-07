
from core.utils.constant import PHELIST_REDUCE
from core.utils.constant import DATASET_TYPE_M, DATASET_TYPE_I, DATASET_TYPE_S, DATASET_TYPE_O
from core.utils.utils import get_all_ancestors_with_dist
from core.reader.hpo_reader import HPOReader
from core.explainer.utils import get_match_impre_noise

import numpy as np
from tqdm import tqdm


def get_no_noise_patient(pa_hpo_list, padis_list, dis_to_hpo_set, hpo_dict, keep_types=None):
	"""
	Returns:
		list: new [hpo_code1, hpo_code2]
		list: new [disCode1, disCode2, ...]
	"""
	rate_tuple = (-np.inf, -np.inf, -np.inf, -np.inf)    # match, impre, noise_spe, noi_oth
	type_to_rank = {DATASET_TYPE_M: 0, DATASET_TYPE_I: 1, DATASET_TYPE_S: 2, DATASET_TYPE_O: 3}
	temp = None
	for dis_code in padis_list:
		mat, imp, noi_spe, noi_oth = get_match_impre_noise(dis_to_hpo_set[dis_code], pa_hpo_list, hpo_dict)
		hpo_len = len(pa_hpo_list)
		matr, impr, noi_sper, noi_othr =len(mat)/hpo_len, len(imp)/hpo_len, len(noi_spe)/hpo_len, len(noi_oth)/hpo_len
		if (matr, impr, -noi_sper, -noi_othr) > rate_tuple: #
			rate_tuple = (matr, impr, -noi_sper, -noi_othr)
			temp = ([mat, imp, noi_spe, noi_oth], [dis_code])
	new_pa_hpo_list = []
	for keepType in keep_types:
		new_pa_hpo_list.extend(temp[0][type_to_rank[keepType]])
	return [new_pa_hpo_list, temp[1]]


def get_no_noise_dataset(dataset, keep_types=None):
	"""
	Args:
		dataset (list): [[hpo_list, dis_list], ...]
		keep_types (list): [type1, type2, ...]; type=DATASET_TYPE_M|DATASET_TYPE_I|DATASET_TYPE_S|DATASET_TYPE_O
	Returns:
		list: new dataset
	"""
	if keep_types is None:
		keep_types = [DATASET_TYPE_M, DATASET_TYPE_I, DATASET_TYPE_S, DATASET_TYPE_O]
	hpo_reader = HPOReader()
	hpo_dict = hpo_reader.get_slice_hpo_dict()
	dis_to_hpo_set = {dis_code: set(hpo_list) for dis_code, hpo_list in hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE).items()}
	dataset = [get_no_noise_patient(pa_hpo_list, padis_list, dis_to_hpo_set, hpo_dict, keep_types) for pa_hpo_list, padis_list in tqdm(dataset)]
	return dataset


def getProbabledis_list(dataset):
	"""
	Returns:
		list: [[disCode1], [disCode2], ...]; length=len(dataset)
	"""
	def check(dataset):
		for _, dis_list in dataset:
			assert len(dis_list) == 1
	dataset = get_no_noise_dataset(dataset)
	check(dataset)
	return [dis_list for hpo_list, dis_list in dataset]


if __name__ == '__main__':
	from core.predict.model_testor import ModelTestor
	mt = ModelTestor()
	mt.load_test_data()
	for data_name, dataset in mt.data.items():
		get_no_noise_dataset(dataset, keep_types=[DATASET_TYPE_M, DATASET_TYPE_I, DATASET_TYPE_S])

