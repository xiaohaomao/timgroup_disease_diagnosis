from collections import Counter
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from core.utils.utils import get_all_ancestors_for_many, get_all_descendents_with_dist, del_obj_list_dup
from core.explainer.utils import get_match_impre_noise_with_dist
from core.utils.constant import ROOT_HPO_CODE, PHELIST_REDUCE
from core.explainer.explainer import Explainer
from core.reader.hpo_reader import HPOReader
from core.reader.hpo_filter_reader import HPOIntegratedDatasetReader


def get_match_impre_noise_for_dataset(dataset, hpo_reader=HPOReader()):
	ave_rate_vec = np.array([0.0, 0.0, 0.0, 0.0])
	dis2hpo = hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
	hpo_dict = hpo_reader.get_slice_hpo_dict()
	valid_pa_num = 0
	for pa in tqdm(dataset):
		pa_hpo_list, pa_dis_list = pa
		pa_hpo_len = len(pa_hpo_list)
		rate_tuple = (np.inf, np.inf, np.inf, np.inf)
		for dis_code in pa_dis_list:
			if dis_code not in dis2hpo:
				continue
			mat, imp, noi_spe, noi_oth = get_match_impre_noise_with_dist(dis2hpo[dis_code], pa_hpo_list, hpo_dict)
			matr, impr, noi_sper, noi_othr = len(mat) / pa_hpo_len, len(imp) / pa_hpo_len, len(noi_spe) / pa_hpo_len, len(noi_oth) / pa_hpo_len
			if noi_othr < rate_tuple[3]: #
				rate_tuple = (matr, impr, noi_sper, noi_othr)
		if rate_tuple[0] == np.inf:
			continue
		ave_rate_vec += np.array(rate_tuple); valid_pa_num += 1
	ave_rate_vec /= valid_pa_num
	return ave_rate_vec.tolist()


class DatasetExplainer(Explainer):
	def __init__(self, hpo_reader=HPOReader()):
		super(DatasetExplainer, self).__init__(hpo_reader)


class UnLabeledDatasetExplainer(DatasetExplainer):
	def __init__(self, upatients, hpo_reader=HPOReader()):
		"""
		Args:
			upatients (list): [[hpo1, hpo2], ...]
		"""
		super(UnLabeledDatasetExplainer, self).__init__(hpo_reader)
		self.upatients = upatients


	def explain(self):
		"""
		Returns:
			dict {
				'PATIENT_NUM': int,
				'AVERAGE_HPO_NUM': int,
				'MAX_HPO_NUM': int,
				'MIN_HPO_NUM': int,
				'AVERAGE_HPO_DEPTH': int,
				'PATIENT_PHE_UNIQUE_NUM': int,
				'HPO_NUM_COUNT': dict
			}
		"""
		d = {
			'PATIENT_NUM':len(self.upatients),
		}
		code_nums = [len(hpo_list) for hpo_list in self.upatients]
		d['MIN_HPO_NUM'], d['MAX_HPO_NUM'] = min(code_nums), max(code_nums)
		d['MEDIAN_HPO_NUM'], d['AVERAGE_HPO_NUM'] = np.median(code_nums), np.mean(code_nums)
		hpo2depth = get_all_descendents_with_dist(ROOT_HPO_CODE, self.hpo_reader.get_slice_hpo_dict())
		d['AVERAGE_HPO_DEPTH'] = np.mean([np.mean([hpo2depth[hpo] for hpo in hpo_list]) for hpo_list in self.upatients])
		d['PATIENT_PHE_UNIQUE_NUM'] = len(del_obj_list_dup(self.upatients, lambda hpo_list: tuple(sorted(hpo_list))))
		d['HPO_NUM_COUNT'] = Counter([len(hpo_list) for hpo_list in self.upatients])
		return d


class LabeledDatasetExplainer(UnLabeledDatasetExplainer):
	def __init__(self, patients, hpo_reader=HPOReader()):

		upatients = [hpo_list for hpo_list, _ in patients]
		super(LabeledDatasetExplainer, self).__init__(upatients, hpo_reader)
		self.patients = patients


	def dis_list_to_key(self, dis_list):
		return tuple(sorted(dis_list))


	def contain_knowledge_code(self, dis_codes, source):
		"""
		Args:
			source (str): 'OMIM' | 'ORPHA' | 'CCRD'
		"""
		for dis_code in dis_codes:
			if dis_code.startswith(source):
				return True
		return False


	def explain(self):
		"""
		Returns:
			dict {
				'PATIENT_NUM': int,
				'AVERAGE_HPO_NUM': int,
				'MAX_HPO_NUM': int,
				'MIN_HPO_NUM': int,
				'AVERAGE_HPO_DEPTH': int,

				'AVERAGE_DIS_NUM': int,
				'MAX_DIS_NUM': int,
				'MIN_DIS_NUM': int,
				'DISEASE_CATEGORY': int,
				'DISEASE_FREQ_MAX': int,
				'DISEASE_FREQ_MIN': int,
			}
		"""
		d = super(LabeledDatasetExplainer, self).explain()
		dis_nums = [len(dis_list) for _, dis_list in self.patients]
		d['MIN_DIS_NUM'], d['MAX_DIS_NUM'], d['AVERAGE_DIS_NUM'] = min(dis_nums), max(dis_nums), np.mean(dis_nums)

		d['PA_WITH_OMIM'] = sum([self.contain_knowledge_code(dis_list, 'OMIM') for _, dis_list in self.patients])
		d['PA_WITH_ORPHA'] = sum([self.contain_knowledge_code(dis_list, 'ORPHA') for _, dis_list in self.patients])
		d['PA_WITH_CCRD'] = sum([self.contain_knowledge_code(dis_list, 'CCRD') for _, dis_list in self.patients])

		dis_count_list = Counter([dis_code for _, dis_list in self.patients for dis_code in dis_list]).most_common()  # [(dis_code, count), ...]
		d['DISEASE_CATEGORY'] = len(dis_count_list)
		d['DISEASE_FREQ_MAX'], d['DISEASE_FREQ_MIN'] = dis_count_list[0][1], dis_count_list[-1][1]
		d['DISEASE_FREQ_MEDIAN'] = np.median([count for dis, count in dis_count_list])

		dis_set_count_list = Counter([self.dis_list_to_key(dis_list) for _, dis_list in self.patients]).most_common()
		d['DISEASE_SET_CATEGORY'] = len(dis_set_count_list)
		d['DISEASE_SET_FREQ_MAX'], d['DISEASE_SET_FREQ_MIN'] = dis_set_count_list[0][1], dis_set_count_list[-1][1]
		d['DISEASE_SET_FREQ_MEDIAN'] = np.median([count for dis, count in dis_set_count_list])


		d['DISEASE_SET_COUNT'] = dis_set_count_list
		d['DISEASE_COUNT_CNS'] = dis_count_list

		mat, imp, noi_spe, noi_oth = get_match_impre_noise_for_dataset(self.patients, hpo_reader=HPOReader())
		d['AVE_EXACT_HPO'] = mat
		d['AVE_GENERAL_HPO'] = imp
		d['AVE_SPECIFIC_HPO'] = noi_spe
		d['AVE_OTHER_HPO'] = noi_oth

		return d


if __name__ == '__main__':
	pass
