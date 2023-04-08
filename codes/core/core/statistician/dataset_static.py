import numpy as np
import os
from tqdm import tqdm
import random
import json
from collections import Counter
from core.utils.constant import RESULT_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, TEST_DATA, get_tune_data_names
from core.utils.constant import DATASET_TYPE_M, DATASET_TYPE_I, DATASET_TYPE_S, DATASET_TYPE_O
from core.utils.utils import get_all_ancestors_with_dist, list_find, count_unique_item, load_save_for_func, get_all_descendents_with_dist
from core.explainer.utils import add_tab, get_match_impre_noise_with_dist, obj2str, obj_to_str_with_max_depth
from core.explainer.explainer import Explainer
from core.reader.hpo_reader import HPOReader
from core.reader.orphanet_reader import OrphanetReader
from core.reader.omim_reader import OMIMReader
from core.predict.model_testor import ModelTestor


def cal_ave_hpo_depth(hpo_list, hpo2depth):
	return np.mean([hpo2depth[hpo] for hpo in hpo_list])


def dataset_static(dataset, out_file=RESULT_PATH+'/dataset_statistic/static.txt'):
	"""
	Args:
		dataset (list): [[hpo_list, dis_list], ...]
	"""
	explainer = Explainer()
	hpo_reader = HPOReader()
	hpo_dict = hpo_reader.get_slice_hpo_dict()
	dis2hpo = {dis_code: set(hpo_list) for dis_code, hpo_list in hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE).items()}

	dis_info = dict(OMIMReader().get_cns_omim(), **OrphanetReader().get_cns_orpha_dict())
	output_str = ''
	ave_rate_vec = np.array([0.0, 0.0, 0.0, 0.0])
	for i in tqdm(range(len(dataset))):
		hpo_list, dis_list = dataset[i]
		rate_tuple = (-np.inf, -np.inf, -np.inf, -np.inf)
		for dis_code in dis_list:
			mat, imp, noi_spe, noi_oth = get_match_impre_noise_with_dist(dis2hpo[dis_code], hpo_list, hpo_dict)
			hpo_len = len(hpo_list)
			true_hpos = list(dis2hpo[dis_code])
			matr, impr, noi_sper, noi_othr =len(mat)/hpo_len, len(imp)/hpo_len, len(noi_spe)/hpo_len, len(noi_oth)/hpo_len
			mat, imp, noi_spe, noi_oth, true_hpos = explainer.add_hpo_info( (mat, imp, noi_spe, noi_oth, true_hpos) )
			output_str += '' \
				'Patient {pa_index}: {dis_code} | {dis_eng} | {dis_cns} \n' \
				'dis_hpos: {true_hpos} \n' \
				'pa_hpos: \n' \
				'match = {match_hpos} \n' \
				'impre = {impre_hpos} \n' \
				'noise_spe = {noise_spe_hpos} \n' \
				'noise_oth = {noise_oth_hpos}  \n' \
				'match_rate impre_rate noise_spe_rate noise_oth_rate: {match_rate} {impre_rate} {noise_spe_rate} {noise_oth_rate} \n' \
				'match_rate impre_rate noise_rate: {match_rate} {impre_rate} {noise_rate}  \n\n'.format(
				pa_index=i, dis_code=dis_code, dis_eng=dis_info.get(dis_code, {}).get('ENG_NAME', ''), dis_cns=dis_info.get(dis_code, {}).get('CNS_NAME', ''),
				true_hpos=true_hpos,
				match_hpos=mat, impre_hpos=imp, noise_spe_hpos=noi_spe, noise_oth_hpos=noi_oth,
				match_rate=matr, impre_rate=impr, noise_spe_rate=noi_sper, noise_oth_rate=noi_othr, noise_rate=noi_sper+noi_othr
			)
			if (matr, impr, -noi_sper, -noi_othr) > rate_tuple: #
				rate_tuple = (matr, impr, -noi_sper, -noi_othr)
		ave_rate_vec += np.array(rate_tuple)
	ave_rate_vec[2:] = -ave_rate_vec[2:]; ave_rate_vec /= len(dataset)
	output_str += 'Average match_rate, impre_rate, noise_spe_rate, noi_oth_rate = {0}\n'.format(ave_rate_vec)
	output_str += 'Average match_rate, impre_rate, noise_rate = {0}\n'.format([ave_rate_vec[0], ave_rate_vec[1], ave_rate_vec[2]+ave_rate_vec[3]])
	output_str += 'Patient Number = {0}\n'.format(len(dataset))
	disease_counter = Counter([dis_code for _, dis_list in dataset for dis_code in dis_list])
	output_str += 'Disease Class Number={0}; Disease appear most={1}; Disease appear least={2}\n'.format(
		len(disease_counter), max(disease_counter.items(), key=lambda item: item[1]), min(disease_counter.items(), key=lambda item: item[1])
	)
	dis_map_rank = hpo_reader.get_dis_map_rank(); DIS_NUM = hpo_reader.get_dis_num()
	ave_dis_rank = np.mean([dis_map_rank[dis_code] for _, dis_list in dataset for dis_code in dis_list])
	output_str += 'Disease Average Rank in dis_list={0}/{1}={2}\n'.format(ave_dis_rank, DIS_NUM, ave_dis_rank/DIS_NUM)
	output_str += 'Average HPO Length = {0}\n'.format( np.mean([len(hpo_list) for hpo_list, _ in dataset]) )
	hpo2depth = get_all_descendents_with_dist('HP:0000001', hpo_dict)
	pa_ave_hpo_depth = np.mean([cal_ave_hpo_depth(hpo_list, hpo2depth) for hpo_list, dis_list in dataset])
	dis_ave_hpo_depth = np.mean([cal_ave_hpo_depth(dis2hpo[dis_code], hpo2depth) for hpo_list, dis_list in dataset for dis_code in dis_list])
	output_str += '' \
		'Average HPO Depth of Patients = {0} \n' \
		'Average HPO Depth of True Disease = {1} \n'.format(pa_ave_hpo_depth, dis_ave_hpo_depth)
	print(output_str, file=open(out_file, 'w'))


def all_dataset_static(folder=None):
	mt = ModelTestor()
	mt.load_test_data()
	folder = RESULT_PATH+'/dataset_statistic' if folder is None else folder
	os.makedirs(folder, exist_ok=True)
	for data_name, dataset in mt.data.items():
		dataset_static(dataset, folder+'/{0}'.format(data_name))


def show_predict_result(dataset, raw_results, out_file, index_to_show_set=None, top_n=20, near_n=2):
	"""
	Args:
		dataset (list): [[hpo_list, dis_list], ...]
		raw_results (list): [[(dis_code, score), ...], ]
		out_file (str)
	"""
	def result_to_str(rank, raw_result, pa_hpos):
		dis_code, score = raw_result[rank]
		dis_hpos = dis2hpo[dis_code]
		mat, imp, noi_spe, noi_oth = get_match_impre_noise_with_dist(dis_hpos, pa_hpos, hpo_dict)
		pa_hpo_len = len(pa_hpos)
		matr, impr, noi_sper, noi_othr =len(mat)/pa_hpo_len, len(imp)/pa_hpo_len, len(noi_spe)/pa_hpo_len, len(noi_oth)/pa_hpo_len
		mat, imp, noi_spe, noi_oth, dis_hpos = explainer.add_hpo_info( (mat, imp, noi_spe, noi_oth, dis_hpos) ) # add Chinese
		return '' \
			'rank={rank}; score={score}; {dis_code} | {dis_cns} \n' \
			'dis_hpos: {dis_hpos} \n' \
			'pa_hpos: \n' \
			'match = {match_hpos} \n' \
			'impre = {impre_hpos} \n' \
			'noise_spe = {noise_spe_hpos} \n' \
			'noise_oth = {noise_oth_hpos}  \n' \
			'match_rate impre_rate noise_spe_rate noise_oth_rate: {match_rate} {impre_rate} {noise_spe_rate} {noise_oth_rate} \n' \
			'match_rate impre_rate noise_rate: {match_rate} {impre_rate} {noise_rate}  \n\n'.format(
				rank=rank+1, score=score,
				dis_code=dis_code, dis_cns=dis_info.get(dis_code, ''),
				dis_hpos=dis_hpos,
				match_hpos=mat, impre_hpos=imp, noise_spe_hpos=noi_spe, noise_oth_hpos=noi_oth,
				match_rate=matr, impre_rate=impr, noise_spe_rate=noi_sper, noise_oth_rate=noi_othr, noise_rate=noi_sper+noi_othr
			)
	explainer = Explainer()
	if index_to_show_set is None:
		index_to_show_set = set(range(len(dataset)))
	hpo_reader = HPOReader()
	hpo_dict = hpo_reader.get_slice_hpo_dict()
	dis2hpo = {dis_code: set(hpo_list) for dis_code, hpo_list in hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE).items()}
	# hpo_to_cns = {hpo: info_dict['CNS_NAME'] for hpo, info_dict in hpo_reader.get_chpo_dict().items()}
	dis_info = dict(explainer.get_omim_to_info(), **explainer.get_orpha_to_info())
	DIS_NUM = hpo_reader.get_dis_num()
	output_str = ''
	# ave_usr = 0.0    # unique score rate
	top_n_cal_usr_array = np.array([5, 10, 20, 50, 100, 200, DIS_NUM])
	ave_usr_array = np.array([0.0] * len(top_n_cal_usr_array))    # unique score rate
	for i in tqdm(range(len(dataset))):
		pa_hpos, dis_list = dataset[i]
		result = raw_results[i]; assert len(result) == DIS_NUM
		usr_array = np.array([
			count_unique_item([tuple(score_item) if isinstance(score_item, np.ndarray) else score_item for dis_code, score_item in result[:n]])
			for n in top_n_cal_usr_array
		]) / top_n_cal_usr_array
		ave_usr_array += usr_array

		if i not in index_to_show_set:   #
			continue
		output_str += '' \
			'======================================================================================================\n' \
			'Patient {pa_index} \n' \
			'Diagnosis: \n' \
			'{pa_diags} \n' \
			'Symptoms: \n' \
			'{pa_hpos} \n' \
			'Unique Score Rate (TopN, Rate) = {usr} \n\n'.format(
			pa_index=i, pa_diags='\n'.join( explainer.add_dis_cns_info(dis_list) ),
			pa_hpos=explainer.add_hpo_info(pa_hpos), usr=list(zip(top_n_cal_usr_array, usr_array))
		)
		output_str += '' \
			'------------------------------------------------------------------------------------------------------\n' \
			'Diagnosis Score: \n'
		show_diag_rank = list_find(result, lambda x: x[0] in set(dis_list)); assert show_diag_rank >= 0
		show_diag_list = [ result[show_diag_rank][0] ]
		for dis_code in show_diag_list:
			rank = list_find(result, lambda item: item[0] == dis_code); assert rank >= 0
			output_str += '' \
				'{pa_diag}; rank={rank} | {pa_diag_cns} \n' \
				'------------------\n'.format(
				pa_diag=dis_code,
				pa_diag_cns=dis_info.get(dis_code, ''), rank=rank+1
			)
			for j in range(max(rank-near_n, 0), min(rank+near_n+1, DIS_NUM)):
				output_str += result_to_str(j, result, pa_hpos)
		output_str += '' \
			'------------------------------------------------------------------------------------------------------\n' \
			'Top {top_n} Score: \n'.format(top_n=top_n)
		for j in range(top_n):
			output_str += result_to_str(j, result, pa_hpos)
	output_str += 'Average Unique Score Rate (TopN, Rate) = {ave_usr} \n'.format( ave_usr=list(zip( top_n_cal_usr_array, ave_usr_array / len(dataset) )) )
	print(output_str, file=open(out_file, 'w'))


def get_index_to_show(dataset, max_ShowNumber, json_path):
	if os.path.exists(json_path):
		return set(json.load(open(json_path)))
	index_list = sorted(random.sample(range(len(dataset)), min(max_ShowNumber, len(dataset))))
	json.dump(index_list, open(json_path, 'w'), indent=2)
	return set(index_list)


def get_real_data_names():
	return ['RAMEDIS', 'HMS', 'MME', 'PC']


def show_model_result(model_name, index_to_show_folder=RESULT_PATH+'/view_result/indexToShow', max_ShowNumber=200, top_n=20, near_n=2, data_names=None):
	print(model_name)
	os.makedirs(index_to_show_folder, exist_ok=True)
	folder = RESULT_PATH+'/view_result/{}'.format(model_name)
	os.makedirs(folder, exist_ok=True)
	mt = ModelTestor(TEST_DATA)
	mt.load_test_data()
	data_names = data_names or get_tune_data_names(TEST_DATA)

	index_to_show_dict = {}
	for data_name, dataset in mt.data.items():
		index_to_show_dict[data_name] = get_index_to_show(
			dataset, max_ShowNumber, index_to_show_folder+'/{0}-{1}.json'.format(data_name, max_ShowNumber)
		)
	for data_name in data_names:
		dataset = mt.get_dataset(data_name)
		raw_results = mt.load_raw_results(model_name, data_name)
		show_predict_result(
			dataset, raw_results, folder+'/{0}-{1}'.format(data_name, max_ShowNumber),
			index_to_show_set=index_to_show_dict[data_name], top_n=top_n, near_n=near_n
		)


def not_anno_hpo_static():
	# all dataset zero
	no_anno_hpo_ints = HPOReader().get_no_anno_hpo_int_set()
	mt = ModelTestor()
	mt.load_test_data()
	for data_name, dataset in mt.data.items():
		allHPONum, all_no_anno_hpo_num = 0, 0
		for hpo_list, dis_list in dataset:
			allHPONum += len(hpo_list)
			all_no_anno_hpo_num += len([hpo for hpo in hpo_list if hpo in no_anno_hpo_ints])
		print('{}: All HPO Number = {}; No Annotation HPO Number = {}({}%)'.format(
			data_name, allHPONum, all_no_anno_hpo_num, all_no_anno_hpo_num/allHPONum))


def patient_num_static(dataset):
	"""
	Args:
		dataset (list): [[hpo_list, dis_list], ...]
	"""
	pass


if __name__ == '__main__':
	pass






