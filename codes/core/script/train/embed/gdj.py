"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""
import os
import numpy as np

from core.node_embed.GDJointEncoder import GDJointEncoder, W_SIGMOID, EUCLIDIAN_DIST2, DIS_SIM_MICA, SIM_TO_DIST_E
from core.node_embed.GDJointEncoder import GDJointConfig, phe_lists_to_embed_mat, COMBINE_HPO_AVG, COMBINE_HPO_IC_WEIGHT
from core.helper.hyper.para_grid_searcher import ParaGridSearcher
from core.explainer.hpo_embedExplainer import hpo_embedExplainer
from core.utils.utils import get_all_descendents_with_dist, get_all_descendents, item_list_to_rank_list, combine_embed
from core.utils.constant import TEST_DATA, RESULT_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, TANH, SIGMOID
from core.reader.hpo_reader import HPOReader
from core.helper.data.data_helper import DataHelper

def train_encoder(d):
	print(d)
	if 'unit-active' in d:
		d['f_units'], d['f_actives'] = d['unit-active']
		del d['unit-active']
	c = GDJointConfig(d)

	encoder_name = 'GDJointEncoder-{}-{}-{}-{}-{}'.format(c.d_combine_hpo, c.phe_list_mode, c.alpha1, c.alpha2, c.co_phe_list_mode)
	encoder = GDJointEncoder(encoder_name=encoder_name)
	encoder.train(c)
	return encoder, c


def get_grid():
	grid = {
		'n_features': [256],
		'unit-active': [([], [])],

		'use_W': [True],
		'w_type': [W_SIGMOID],
		'beta': [0.5], # 0.3

		'alpha1': [1.0],   # 0.1, 0.5
		'use_dis_dist': [True],
		'dis_sim_type': [DIS_SIM_MICA],
		'sim_to_dist': [SIM_TO_DIST_E],
		'embed_dist': [EUCLIDIAN_DIST2],
		'dis_dist_min': [0.1],   # 0.0
		'd_combine_hpo':[COMBINE_HPO_AVG],
		'phe_list_mode':[PHELIST_REDUCE],

		'alpha2': [1.0 , 2.0, 3.0],
		'co_phe_list_mode': [PHELIST_REDUCE, PHELIST_ANCESTOR],
		'co_W_type': [W_SIGMOID],
		'co_W_beta': [1.0],

	}
	return grid


def run(cpu_use=1):
	grid = get_grid()
	para_iterator = ParaGridSearcher(grid).iterator()
	if cpu_use == 1:
		for d in para_iterator:
			encoder, c = train_encoder(d)
			draw(encoder, c)


def draw(encoder, c):
	hpo_embed = encoder.get_embed()
	model_fig_folder = encoder.SAVE_FOLDER + '/tsne'

	print('draw hpo embedding...')
	folder = model_fig_folder + '/hpo_embed'; os.makedirs(folder, exist_ok=True)
	draw_hpo_embed(hpo_embed, folder)

	print('draw disease embedding...')
	folder = model_fig_folder + '/DisEmbed'; os.makedirs(folder, exist_ok=True)
	draw_dis_embed(hpo_embed, c.phe_list_mode, c.d_combine_hpo, folder)
	draw_dataset_dis_embed(hpo_embed, c.phe_list_mode, c.d_combine_hpo, folder)


def get_show_tree_roots():
	return [
		'HP:0000079',   # 泌尿系统异常
		'HP:0000924',   # 骨骼系统异常
		'HP:0001939',   # 代谢紊乱/稳态失衡
	]


def get_show_dis_codes():
	dis_codes = [
		'OMIM:615067',
		'OMIM:613808',
		'OMIM:614679',

		'OMIM:613108',
		'OMIM:212050',

		'OMIM:602433',
		'OMIM:612069',
	]
	return dis_codes


def get_show_dataset_dis():
	return {
		'RAMEDIS': [
			'OMIM:237300',  #
			'OMIM:261600',  #
			'OMIM:248600',
			'OMIM:236792',  #
		],
		'PC': [
			'OMIM:136140',  #
			'OMIM:610536',  #
			'OMIM:616789',  # MENTAL RETARDATION AND DISTINCTIVE FACIAL FEATURES WITH OR WITHOUT CARDIAC DEFECTS; MRFACD
			'OMIM:301835',  #
		]
	}


def draw_hpo_embed(hpo_embed, folder):
	embed_mat = hpo_embed
	hpo_reader = HPOReader()
	explainer = hpo_embedExplainer(embed_mat, hpo_reader=hpo_reader)

	tsne_X = explainer.cal_tsne()
	root_hpos = get_show_tree_roots()
	for rootHPO in root_hpos:
		figpath = folder+'/{}.png'.format(explainer.add_hpo_info(rootHPO).replace(':', '.').replace('/', '.'))
		explainer.draw_hpo_tree(figpath, [rootHPO], tsne_X)

	dis_to_hpo_list = hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
	show_dis_codes = get_show_dis_codes()
	for dis_code in show_dis_codes:
		dis_name = explainer.add_dis_cns_info(dis_code).replace(':', '.').replace('/', '.')
		figpath = folder+'/{}.png'.format(dis_name)
		explainer.draw_dis_hpo(figpath, [dis_to_hpo_list[dis_code]], [dis_name], tsne_X)


def draw_dis_embed(hpo_embed_mat, phe_list_mode, d_combine_hpo, folder):
	hpo_reader = HPOReader()
	dh = DataHelper(hpo_reader)
	hpo_int_lists, lb_list = dh.get_train_raw_Xy(phe_list_mode)

	dis_embed_mat = phe_lists_to_embed_mat(hpo_embed_mat, hpo_int_lists, d_combine_hpo)
	explainer = hpo_embedExplainer(dis_embed_mat, hpo_reader)
	tsne_X = explainer.cal_tsne()

	dis_map_rank = hpo_reader.get_dis_map_rank()
	id2lb = {dis_map_rank[dis_code]: explainer.add_dis_cns_info(dis_code) for dis_code in get_show_dis_codes()}
	lb_to_size = {lb: 150 for lb in set(id2lb.values())}
	figpath = folder+'/{}-DisEmbed.png'.format(d_combine_hpo)
	explainer.draw_tsne(figpath, tsne_X, id2label=id2lb, lb_to_size=lb_to_size, add_text=True, embed_type='DIS')


def draw_dataset_dis_embed(hpo_embed_mat, phe_list_mode, d_combine_hpo, folder):
	hpo_reader = HPOReader()
	dh = DataHelper(hpo_reader)

	dataset_dis = get_show_dataset_dis()
	hpo_int_lists, lb_list = dh.get_train_raw_Xy(phe_list_mode)
	data_range = {}; offset = len(hpo_int_lists)  # {dname: (begin, end)}
	dis_map_rank, hpo_map_rank = hpo_reader.get_dis_map_rank(), hpo_reader.get_hpo_map_rank()
	for dname, _ in dataset_dis.items():
		pHPOLists, pdis_lists = zip(*dh.get_dataset(dname, TEST_DATA))
		data_range[dname] = (offset, offset+len(pHPOLists)); offset += len(pHPOLists)
		hpo_int_lists.extend([item_list_to_rank_list(hpo_list, hpo_map_rank) for hpo_list in pHPOLists])
		lb_list.extend([dis_map_rank[pdis_list[0]] for pdis_list in pdis_lists])

	dis_embed_mat = phe_lists_to_embed_mat(hpo_embed_mat, hpo_int_lists, d_combine_hpo)
	explainer = hpo_embedExplainer(dis_embed_mat, hpo_reader)
	tsne_X = explainer.cal_tsne()
	lb_ary = np.array(lb_list)
	dis_list = hpo_reader.get_dis_list()
	for dname, showdis_list in dataset_dis.items():
		for dis_code in showdis_list:
			id2label = {}
			dis_id = dis_map_rank[dis_code]; dis_name = explainer.add_dis_cns_info(dis_list[dis_id])
			id2label[dis_id] = 'dis-' + dis_name
			b, e = data_range[dname]
			pa_ids = np.where(lb_ary[b: e] == dis_id)[0] + b
			pat_names = ['{}-pat{}-{}'.format(dname, paId-b, dis_name) for paId in pa_ids]
			id2label.update({paId: patName for paId, patName in zip(pa_ids, pat_names)})
			lb_order = [id2label[dis_id]] + pat_names
			lb_to_size = {lb: 150 for lb in set(id2label.values())}
			lb2style = {lb: 'o' for lb in set(id2label.values())}; lb2style[id2label[dis_id]] = 's'
			figpath = folder + '/{}-{}-{}.png'.format(d_combine_hpo, dname, dis_name.replace(':', '.').replace('/', '.'))
			explainer.draw_tsne(figpath, tsne_X, id2label, label_order=lb_order, lb_to_size=lb_to_size, lb2style=lb2style)


if __name__ == '__main__':
	pass
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	run()   #

