
import os
import pickle
import json
import numpy as np
from scipy.sparse import csr_matrix
import heapq
from multiprocessing import Pool
from tqdm import tqdm
from core.utils.constant import DATA_PATH, NPY_FILE_FORMAT, JSON_FILE_FORMAT, PHELIST_ANCESTOR, PHELIST_REDUCE, GENE_ANNOTATION, DISEASE_ANNOTATION
from core.utils.utils import check_load_save, pagerank, item_list_to_rank_list, get_all_ancestors_for_many, delete_redundacy, ret_same
from core.reader.hpo_reader import HPOReader


class PageRankNoiseReductor(object):
	def __init__(self, hpo_reader=HPOReader(), SM=0.02):
		self.name = 'PagerankNoiseReductor'
		self.hpo_reader = hpo_reader

		self.SMOOTH_VALUE = SM
		self.DISEASE_PTM_NPY = DATA_PATH + '/preprocess/DiseaseJaccardPTM_SM{0}.npy'.format(self.SMOOTH_VALUE)
		self.GENE_PTM_NPY = DATA_PATH + '/preprocess/GeneJaccardPTM_SM{0}.npy'.format(self.SMOOTH_VALUE)
		self.disease_ptm_mat = None # probability transfer matrix; np.array; shape=[hpo_num, hpo_num]
		self.gene_ptm_mat = None  # probability transfer matrix; np.array; shape=[hpo_num, hpo_num]

		self.DISTOHPO_JSON_PATH = DATA_PATH + '/preprocess/DisToHPOPGReduct_SM{0}.json'.format(self.SMOOTH_VALUE)
		self.DISTOHPO_DIS_ANNO_JSON_PATH = DATA_PATH + '/preprocess/DisToHPOPGReduct_DisAnno_SM{0}.json'.format(self.SMOOTH_VALUE)
		self.dis2hpo = None
		self.dis_to_hpo_dis_anno = None


	def cal_jaccard_ptm(self, anno_lists):

		ANNO_ITEM_NUM = len(anno_lists)
		row, col, data = [], [], []
		hpo_map_rank, HPO_CODE_NUMBER = self.hpo_reader.get_hpo_map_rank(), self.hpo_reader.get_hpo_num()
		for i in range(ANNO_ITEM_NUM):
			rank_list = [hpo_map_rank[hpo_code] for hpo_code in anno_lists[i]]
			row.extend([i]*len(rank_list))
			col.extend(rank_list)
			data.extend([1]*len(rank_list))
		anno_vec_mat = csr_matrix((data, (row, col)), shape=(ANNO_ITEM_NUM, HPO_CODE_NUMBER))    # shape=(annoNum, hpo_num)
		intersect_mat = anno_vec_mat.transpose() * anno_vec_mat  # shape=(hpo_num, hpo_num)
		anno_num_mat = anno_vec_mat.sum(axis=0)  # np.matrix; shape=(1, hpo_num)
		ptm = np.array(intersect_mat / (-intersect_mat + anno_num_mat + anno_num_mat.T))   # np.matrix; shape=(hpo_num, hpo_num)
		ptm[ptm==0] = self.SMOOTH_VALUE #
		ptm[np.isnan(ptm)] = self.SMOOTH_VALUE  #
		ptm[range(ptm.shape[0]), range(ptm.shape[0])] = 1.0
		return ptm


	@check_load_save('disease_ptm_mat', 'DISEASE_PTM_NPY', NPY_FILE_FORMAT)
	def get_jaccard_ptm_using_dis_anno(self):
		"""calculating Probability transfer matrix using disease annotation
		"""
		dis_list = self.hpo_reader.get_dis_list()
		dis2hpo = self.hpo_reader.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
		anno_lists = [dis2hpo[dis] for dis in dis_list]
		return self.cal_jaccard_ptm(anno_lists)


	@check_load_save('gene_ptm_mat', 'GENE_PTM_NPY', NPY_FILE_FORMAT)
	def get_jaccard_ptm_using_gene_anno(self):
		"""calculating Probability transfer matrix using gene annotation
		"""
		gene_list = self.hpo_reader.get_gene_list()
		gene2hpo = self.hpo_reader.get_gene_to_hpo_dict(PHELIST_ANCESTOR) #
		anno_list = [gene2hpo[gene] for gene in gene_list]
		return self.cal_jaccard_ptm(anno_list)


	def reduct_noise(self, phe_list, keep_k_func=ret_same, order=False, anno_used=GENE_ANNOTATION):

		keep_k = keep_k_func(len(phe_list))
		if len(phe_list) <= keep_k and not order:
			return phe_list
		hpo_map_rank, hpo_list = self.hpo_reader.get_hpo_map_rank(), self.hpo_reader.get_hpo_list()
		rank_list = item_list_to_rank_list(phe_list, hpo_map_rank)
		if anno_used == GENE_ANNOTATION:
			M = self.get_jaccard_ptm_using_gene_anno()[rank_list, :][:, rank_list]
		else:
			M = self.get_jaccard_ptm_using_dis_anno()[rank_list, :][:, rank_list]
		M = M / M.sum(axis=0)
		v = pagerank(M); assert np.sum(np.isnan(v)) == 0
		result = heapq.nlargest(keep_k, [(v[i], rank_list[i]) for i in range(len(rank_list))])  # [(probability, hpo_rank), ...]
		# print([(hpo_list[result[i][1]],result[i][0]) for i in range(len(result))])
		return [hpo_list[hpo_rank] for _, hpo_rank in result], [prob for prob, hpo_rank in result]


	def reduct_noise_multi_wrap(self, paras):
		return self.reduct_noise(paras[0], paras[1], paras[2], paras[3])


	def reduct_noise_for_many(self, phe_lists, keep_k_func=ret_same, order=False, anno_used=GENE_ANNOTATION):
		"""
		Args:
			phe_lists (list): [[hpo_code1, ...], ...]
			keep_k_func (func)
		Returns:
			list: [[hpo_code1, ...], ...]
			list: [[prob1, ...], ...]
		"""
		ret_phe_lists, ret_prob_lists = [], []
		paras = [(phe_lists[i], keep_k_func, order, anno_used) for i in range(len(phe_lists))]
		with Pool() as pool:
			for phe_list, prob_list in pool.imap(self.reduct_noise_multi_wrap, paras, chunksize=200):
				ret_phe_lists.append(phe_list)
				ret_prob_lists.append(prob_list)
		return ret_phe_lists, ret_prob_lists


	@check_load_save('dis2hpo', 'DISTOHPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_dis_to_hpo(self):
		dis2hpo = self.hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
		dis_list, phe_lists = list(zip(*dis2hpo.items()))
		phe_lists, _ = self.reduct_noise_for_many(phe_lists, order=True)
		return {dis_list[i]: phe_lists[i] for i in range(len(dis_list))}


	@check_load_save('dis_to_hpo_dis_anno', 'DISTOHPO_DIS_ANNO_JSON_PATH', JSON_FILE_FORMAT)
	def get_dis_to_hpo_dis_anno(self):
		dis2hpo = self.hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
		dis_list, phe_lists = list(zip(*dis2hpo.items()))
		phe_lists, _ = self.reduct_noise_for_many(phe_lists, order=True, anno_used=DISEASE_ANNOTATION)
		return {dis_list[i]: phe_lists[i] for i in range(len(dis_list))}


	def get_dis_to_hpo_noise_reduct(self, phe_list_mode=PHELIST_REDUCE, keep_k_func=ret_same, anno_used=GENE_ANNOTATION):
		dis2hpo = self.get_dis_to_hpo() if anno_used == GENE_ANNOTATION else self.get_dis_to_hpo_dis_anno()
		dis2hpo = {dis_code: phe_list[:keep_k_func(len(phe_list))] for dis_code, phe_list in dis2hpo.items()}
		if phe_list_mode == PHELIST_ANCESTOR:
			hpo_dict = self.hpo_reader.get_hpo_dict()
			dis2hpo = {dis_code: list(get_all_ancestors_for_many(phe_list, hpo_dict)) for dis_code, phe_list in dis2hpo.items()}
		return dis2hpo


def cal_noise_reduct_rate_delete_true_n(origin_phe_lists, ordered_noise_phe_lists):
	assert len(origin_phe_lists) == len(ordered_noise_phe_lists)
	DATA_SIZE = len(origin_phe_lists)
	noise_reduct_rate_sum = 0
	for i, ordered_noise_phe_list in enumerate(ordered_noise_phe_lists):
		origin_phe_set, noise_phe_set = set(origin_phe_lists[i]), set(ordered_noise_phe_list)
		noise_hpos = noise_phe_set - origin_phe_set
		origin_hpo_num, noise_hpo_num = len(origin_phe_set), len(noise_hpos)
		if noise_hpo_num == 0:
			continue
		noise_reduct_rate = sum([1 for hpo in ordered_noise_phe_list[-noise_hpo_num: ] if hpo in noise_hpos]) / noise_hpo_num
		noise_reduct_rate_sum += noise_reduct_rate
	return noise_reduct_rate_sum / DATA_SIZE


def cal_noise_reduct_rate_delete_n(origin_phe_lists, ordered_noise_phe_lists, n):
	assert len(origin_phe_lists) == len(ordered_noise_phe_lists)
	DATA_SIZE = len(origin_phe_lists)
	noise_reduct_rate_sum = 0
	for i in range(DATA_SIZE):
		noise_hpos = set(ordered_noise_phe_lists[i]) - set(origin_phe_lists[i])
		noise_reduct_rate = sum([1 for hpo in ordered_noise_phe_lists[i][-n: ] if hpo in noise_hpos]) / n
		noise_reduct_rate_sum += noise_reduct_rate
	return noise_reduct_rate_sum / DATA_SIZE


def cal_diff_rank_ratio(dataset, nr, hpo_reader):
	"""
	Args:
		dataset (list): [[hpo_list, dis_list], ...]
	"""
	def item_to_value(ll, d):
		return [d[item[0]] if isinstance(item, tuple) else d[item] for item in ll]
	from core.explainer.utils import get_match_impre_noise
	hpo_dict = hpo_reader.get_hpo_dict()
	dis2hpo = hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
	nr_pa_hposList, nr_pa_hpo_pg_probs_list = nr.reduct_noise_for_many([pa_hpos for pa_hpos, _ in dataset], order=True, anno_used=DISEASE_ANNOTATION)
	diff_ratios = []
	output_str = ''
	for i in range(len(dataset)):
		nr_pa_hpos = nr_pa_hposList[i]; nr_pa_hpo_pg_probs = nr_pa_hpo_pg_probs_list[i]
		hpo_to_pg_prob = {hpo: pgProb for hpo, pgProb in zip(nr_pa_hpos, nr_pa_hpo_pg_probs)}
		dis_hpos = dis2hpo[dataset[i][1][0]]
		mat, imp, noi_spe, noi_oth = get_match_impre_noise(dis_hpos, nr_pa_hpos, hpo_dict)
		noise_set = set(noi_oth)
		useful_rank = np.mean([i+1 for i in range(len(nr_pa_hpos)) if nr_pa_hpos[i] not in noise_set])
		noise_rank = np.mean([i+1 for i in range(len(nr_pa_hpos)) if nr_pa_hpos[i] in noise_set])
		drr = 0.0 if np.isnan(useful_rank) or np.isnan(noise_rank) else (noise_rank - useful_rank) / len(nr_pa_hpos)
		diff_ratios.append(drr)
		output_str += '' \
			'Patient {pid}: \n' \
			'{mat} {imp} {noi_spe} {noi_oth} \n' \
			'{mat_P} {imp_p} {noi_spe_p} {noi_oth_p} \n'.format(
			pid=i, mat=mat, imp=imp, noi_spe=noi_spe, noi_oth=noi_oth, mat_P=item_to_value(mat, hpo_to_pg_prob),
			imp_p=item_to_value(imp, hpo_to_pg_prob), noi_spe_p=item_to_value(noi_spe, hpo_to_pg_prob), noi_oth_p=item_to_value(noi_oth, hpo_to_pg_prob)
		)
	return np.mean(diff_ratios), output_str


if __name__ == '__main__':
	hpo_reader = HPOReader()
	nr = PageRankNoiseReductor(HPOReader(), SM=0.02)

	from core.predict.model_testor import ModelTestor
	from core.utils.constant import RESULT_PATH
	mt = ModelTestor()
	data_names = ['MME_43', 'PC_174', 'DEC_SNV_DIS_155', 'SIM_NOISE']
	mt.load_test_data(data_names)
	output_str = 'DiffRankRatio: \n'
	drr_str = ''
	for sm in tqdm([0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]):
		nr = PageRankNoiseReductor(HPOReader(), SM=sm)
		output_str += 'SM={sm}\t'.format(sm=sm)
		for data_name in data_names:
			diffRankRatio, sub_drr_str = cal_diff_rank_ratio(mt.data[data_name], nr, hpo_reader)

			output_str += '{data_name} {drr}; '.format(sm=sm, data_name=data_name, drr=diffRankRatio)
			drr_str += '' \
				'==========================================\n' \
				'SM={sm} {data_name} {drr} \n' \
				'{sub_drr_str} \n'.format(sm=sm, data_name=data_name, drr=diffRankRatio, sub_drr_str=sub_drr_str)
		output_str += '\n'
	folder = RESULT_PATH + '/PageRank'; os.makedirs(folder, exist_ok=True)
	print(drr_str, output_str, file=open(folder+'/DiffRankRatio_DISEASE_ANNOTATION', 'w'))
