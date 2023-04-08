import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import json

from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_descendents_with_dist, get_all_descendents, get_all_ancestors_for_many, slice_list_with_rm_set, get_logger
from core.utils.utils import dict_list_add, data_to_01_matrix
from core.utils.constant import LOG_PATH, RESULT_PATH, PHELIST_ANCESTOR, PHELIST_REDUCE, RESULT_FIG_PATH, PHELIST_ANCESTOR_DUP
from core.reader.omim_reader import OMIMReader
from core.reader.orphanet_reader import OrphanetReader
from core.draw.simpledraw import simple_dist_plot, simple_line_plot
from core.explainer.explainer import Explainer
from core.predict.calculator.ic_calculator import get_dis_IC_dict, get_hpo_IC_dict, get_dis_IC_vec

def statitic1():
	reader = HPOReader()
	hpo_dict = reader.get_hpo_dict()
	dis2hpo = reader.get_dis_to_hpo_dict()
	logger = get_logger('HPO_STATITIC', log_path=LOG_PATH + '/HPO_STATITIC')

	logger.info('HPO Num (All): %d' % (len(hpo_dict),))
	phe_set = get_all_descendents('HP:0000118', hpo_dict)
	logger.info('HPO Num (HP:0000118, Phenotypic abnormality): %d' % (len(phe_set),))
	logger.info('HPO Num (HP:0012823, Clinical modifier): %d' % (len(get_all_descendents('HP:0012823', hpo_dict)),))
	logger.info('HPO Num (HP:0000005, Mode of inheritance): %d' % (len(get_all_descendents('HP:0000005', hpo_dict)),))
	logger.info('HPO Num (HP:0040279, Frequency): %d' % (len(get_all_descendents('HP:0040279', hpo_dict)),))
	logger.info('HPO Num (HP:0040006, Mortality/Aging): %d' % (len(get_all_descendents('HP:0040006', hpo_dict)),))
	logger.info('Disease Num: %d' % (len(dis2hpo),))
	logger.info('OMIM Disease Num: %d' % (np.sum([dis_code.startswith('OMIM') for dis_code in dis2hpo.keys()]),))
	logger.info('DECIPHER Disease Num: %d' % (np.sum([dis_code.startswith('DECIPHER') for dis_code in dis2hpo.keys()]),))
	logger.info('ORPHA Disease Num: %d' % (np.sum([dis_code.startswith('ORPHA') for dis_code in dis2hpo.keys()]),))
	hpo_ana_set = set()
	for hpo_list in dis2hpo.values():
		hpo_ana_set.update(hpo_list)
	logger.info('HPO Num (In Anatation): %d' % (len(hpo_ana_set),))

	hpo_ana_set = get_all_ancestors_for_many(hpo_ana_set, hpo_dict)
	inter_sect_set = phe_set & hpo_ana_set
	logger.info('HPO Num (In Anatation_Exten): %d' % (len(hpo_ana_set),))
	logger.info('HPO Num (In Anatation_Exten & In Phenotypic abnormality): %d' % (len(inter_sect_set),))
	logger.info('HPO Num (In Anatation_Exten & NOT In Phenotypic abnormality): %d, %s' % (len(hpo_ana_set) - len(inter_sect_set), hpo_ana_set-inter_sect_set))
	logger.info('HPO Num (NOT In Anatation_Exten & In Phenotypic abnormality): %d' % (len(phe_set) - len(inter_sect_set),))

	logger.info('Ave HPO Num of Dis: %d' % (np.mean([len(hpo_list) for hpo_list in dis2hpo.values()]),))
	logger.info('Ave HPO Num (Extend) of Dis: %d' % (np.mean([len(get_all_ancestors_for_many(hpo_list, reader.get_hpo_dict())) for hpo_list in dis2hpo.values()]),))

	counter = Counter()
	shortes_dis_to_root = get_all_descendents_with_dist('HP:0000001', hpo_dict)
	for hpo in shortes_dis_to_root:
		if not hpo_dict[hpo].get('CHILD', []):
			counter[shortes_dis_to_root[hpo]] += 1
	leaf_total = sum(counter.values())
	for distance, leafNum in counter.items():
		logger.info('Leaf Depth: %d %f%%' % (distance, leafNum/leaf_total*100))
	logger.info('Average Leaf depth: %d' % (sum([distance*leafNum for distance, leafNum in counter.items()])/leaf_total,))
	logger.info('Max Shortest depth: %d' % (max(shortes_dis_to_root.values()), ))

	counter = Counter()
	for dis, hpo_list in tqdm(dis2hpo.items()):
		ances_count = 0
		for hpo in hpo_list:
			if hpo in get_all_ancestors_for_many(slice_list_with_rm_set(hpo_list, [hpo]), hpo_dict):
				ances_count += 1
		if ances_count != 0:
			counter[dis] += ances_count
	logger.info('Ancestor Detect: Redundancy Disease = %d/%d' % (len(counter), len(dis2hpo)))
	logger.info('Ancestor Detect: %s' % (str(counter,)))


	dis_hpo_count, dis_hpo_leaf_count = 0, 0
	for dis_code, phe_list in dis2hpo.items():
		dis_hpo_count += len(phe_list)
		for hpo in phe_list:
			if len(hpo_dict[hpo].get('CHILD', [])) == 0:
				dis_hpo_leaf_count += 1
	logger.info('dis_hpo_leaf_count/dis_hpo_count=%d/%d=%f; ' % (dis_hpo_leaf_count, dis_hpo_count, dis_hpo_leaf_count/dis_hpo_count))


def show_dis_with_khpos(k=10):
	"""get disease with HPO less than k
	"""
	folder = RESULT_PATH+os.sep+'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	reader = HPOReader()
	dis2hpo = reader.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
	hpo_num_to_dis_list = {}
	for dis, hpo_list in dis2hpo.items():
		if len(hpo_list) <= k:
			dict_list_add(len(hpo_list), dis, hpo_num_to_dis_list)
	explainer = Explainer()
	s = ''
	for i in range(k+1):
		s += '\n==========================================================================================================\n' \
			'HPO Number = {hpo_num}; Found {dis_num} Disease\n'.format(hpo_num=i, dis_num=len(hpo_num_to_dis_list.get(i, [])))
		for dis_code in hpo_num_to_dis_list.get(i, []):
			s += '{dis}: {HPOs}\n'.format(dis=explainer.add_dis_cns_info(dis_code), HPOs=explainer.add_hpo_info(dis2hpo[dis_code]))
	print(s, file=open(folder+os.sep+'showDisWith{}HPOs'.format(k), 'w'))


def show_dis_with_hpos():
	folder = RESULT_PATH+os.sep+'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	explainer = Explainer()
	reader = HPOReader()
	dis_hpo_list = [(dis, hpos) for dis, hpos in reader.get_dis_to_hpo_dict(PHELIST_ANCESTOR).items()]
	dis_hpo_list = sorted(dis_hpo_list, key=lambda item: len(item[1]))
	shortes_dis_to_root = get_all_descendents_with_dist('HP:0000001', reader.hpo_dict)
	step = 6
	s = ''
	for dis_code, hpo_list in tqdm(dis_hpo_list):
		distance_to_dis = {}
		s += '\n========================================\n' \
		     '{dis}; HPO Number = {hpo_num}:\n'.format(dis=explainer.add_dis_cns_info(dis_code), hpo_num = len(hpo_list))
		for hpo in hpo_list:
			dict_list_add(shortes_dis_to_root[hpo], hpo, distance_to_dis)
		distance_to_dis = explainer.add_hpo_info(distance_to_dis)
		for depth in sorted(distance_to_dis.keys()):
			sub_hpo_list = distance_to_dis[depth]
			s += 'Depth = {}:\n'.format(depth)
			for i in range(0, len(sub_hpo_list), step):
				s += '{}\n'.format(sub_hpo_list[i: i+step])
	print(s, file=open(folder+os.sep+'AllDiseaseWithHPONum', 'w'))


def show_dis_with_hpo_IC():
	folder = RESULT_PATH + os.sep + 'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	explainer = Explainer()
	reader = HPOReader()

	hpo_IC_dict = get_hpo_IC_dict(reader)
	dis_IC_dict = get_dis_IC_dict(reader, PHELIST_ANCESTOR)
	dis_to_hpo_list = reader.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
	dis_IC_list = sorted(dis_IC_dict.items(), key=lambda item: item[1])

	shortes_dis_to_root = get_all_descendents_with_dist('HP:0000001', reader.hpo_dict)
	step = 6
	s = ''
	for dis_code, IC in tqdm(dis_IC_list):
		hpo_list = dis_to_hpo_list[dis_code]
		distance_to_dis = {}
		s += '\n========================================\n' \
		     '{dis}; IC = {IC}; HPO Number = {hpo_num}:\n'.format(dis=explainer.add_dis_cns_info(dis_code), IC=IC, hpo_num=len(hpo_list))
		for hpo in hpo_list:
			dict_list_add(shortes_dis_to_root[hpo], (hpo, hpo_IC_dict[hpo]), distance_to_dis)
		distance_to_dis = explainer.add_hpo_info(distance_to_dis)
		for depth in sorted(distance_to_dis.keys()):
			sub_hpo_list = distance_to_dis[depth]
			s += 'Depth = {}:\n'.format(depth)
			for i in range(0, len(sub_hpo_list), step):
				s += '{}\n'.format(sub_hpo_list[i: i + step])
	print(s, file=open(folder + os.sep + 'AllDiseaseWithIC', 'w'))


def draw_hpo_num_geq_disease(phe_list_mode, x_max=51):
	folder = RESULT_PATH + os.sep + 'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	hpo_reader = HPOReader()

	counter = Counter([len(hpo_list) for _, hpo_list in hpo_reader.get_dis_to_hpo_dict(phe_list_mode).items()])
	max_hpo_num = max(counter.keys())
	x_list = list(range(max_hpo_num, -1, -1))
	hpo_num_sum = 0
	y_list = []
	for x in x_list:
		hpo_num_sum += counter.get(x, 0)
		y_list.append(hpo_num_sum)
	json.dump([(x, y) for x, y in zip(x_list, y_list)], open(folder+'/HPONumGeqDisease-{}.json'.format(phe_list_mode), 'w'), indent=2)
	x_list, y_list = x_list[-x_max:], y_list[-x_max:]
	simple_line_plot(folder+'/HPONumGeqDisease-{}'.format(phe_list_mode), x_list, y_list, 'HPO >= N', 'Disease Number')


def draw_dis_IC_geq_disease(phe_list_mode, max_IC=None, bins=50):
	folder = RESULT_PATH + os.sep + 'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	hpo_reader = HPOReader()

	s_dis_IC_vec = get_dis_IC_vec(hpo_reader, phe_list_mode)
	s_dis_IC_vec.sort()
	max_IC = max_IC or s_dis_IC_vec.max()
	x_array = np.linspace(0, max_IC, bins)
	y_array = s_dis_IC_vec.shape[0] - np.searchsorted(s_dis_IC_vec, x_array)
	json.dump([(x, y) for x, y in zip([float(x) for x in x_array], [int(y) for y in y_array])], open(folder + '/DisICGeqDisease-{}.json'.format(phe_list_mode), 'w'), indent=2)
	simple_line_plot(folder + '/DisICGeqDisease-{}'.format(phe_list_mode), x_array, y_array, 'IC >= X', 'Disease Number')


def draw_hpo_num_per_disease():
	def run(phe_type, figname, title, bins, figsize=(20, 10)):
		hpo_reader = HPOReader()
		dis_to_hpo_dict = hpo_reader.get_dis_to_hpo_dict(phe_type)
		total_hpo = sum([len(hpo_list) for hpo_list in dis_to_hpo_dict.values()])
		print('Total HPO Number = {}; Average HPO Number per Disease = {}'.format(total_hpo, total_hpo/len(dis_to_hpo_dict)))
		simple_dist_plot(
			RESULT_PATH+os.sep+'hpo_statistic'+os.sep+figname,
			[len(hpo_list) for hpo_list in dis_to_hpo_dict.values()],
			bins,
			x_label='HPO Number',
			title=title,
			figsize=figsize
		)
	run(PHELIST_REDUCE, '{}.jpeg'.format('hpoNumPerDisease.reduce'), 'HPO Number(reduce) of Disease', 175)


def show_dis_cns_list():
	folder = RESULT_PATH + os.sep + 'hpo_statistic'; os.makedirs(folder, exist_ok=True)
	dis_list = Explainer().add_dis_cns_info(HPOReader().get_dis_list())
	json.dump(dis_list, open(folder+os.sep+'dis_cnsList.json', 'w'), indent=2, ensure_ascii=False)


def statistic5():
	hpo_dict = HPOReader().get_slice_hpo_dict()
	counter = Counter([len(info.get('IS_A', [])) for hpo, info in hpo_dict.items()])
	print(counter)


if __name__ == '__main__':
	pass



