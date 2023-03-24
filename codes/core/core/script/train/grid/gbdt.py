

from core.predict.model_testor import ModelTestor
from core.predict.ml_model.GBDTModel import generate_model, GBDTConfig
from core.reader.hpo_reader import HPOReader
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, TRAIN_MODE
from core.utils.constant import PHELIST_ANCESTOR, VEC_COMBINE_MEAN, VEC_COMBINE_MAX, PHELIST_REDUCE, PHELIST_ANCESTOR_DUP
from core.utils.utils import get_logger
from core.helper.data.data_helper import DataHelper

import os
import itertools


def process(para):
	rf_config, model_name = para

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	model.train(raw_X, y_, sw, rf_config, None, save_model=False)

	return model


def train_script():
	# tree_num_list = [100, 500, 1000, 5000, 10000]
	# max_leaves_list = [2**5-1, 2**6-1, 2**7-1]
	tree_num_list = [100, 500, 1000]
	max_leaves_list = [2**6-1, 2**7-1]    #
	bagging_frac_list = [0.8]
	feature_frac_list = [0.8]

	paras = []
	for tree_num, max_leaves, bagfrac, feafrac in itertools.product(tree_num_list, max_leaves_list, bagging_frac_list, feature_frac_list):
		gbdt_config = GBDTConfig()
		gbdt_config.tree_num = tree_num
		gbdt_config.max_leaves = max_leaves
		gbdt_config.bagging_frac = bagfrac
		gbdt_config.feature_frac = feafrac
		model_name = 'GBDTModel_01_Ances_tn{}_leaf{}_bag{}_fea{}'.format(tree_num, max_leaves, bagfrac, feafrac)
		paras.append((gbdt_config, model_name))

	mt = ModelTestor()
	mt.load_test_data()
	for para in paras:
		model = process(para)
		for data_name in mt.data_names:
			mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)



# ==========================================================================================
def get_deepwalk_embed_paras():
	from core.node_embed.DeepwalkEncoder import get_embed
	win_list = [1, 2, 4]
	num_walk_list = [200]
	embed_size_list = [128]
	paras = []
	for win, numWalk, embed_size in itertools.product(win_list, num_walk_list, embed_size_list):
		encoder_name = 'DeepwalkEncoder_win{}_numwalks{}_embed{}'.format(win, numWalk, embed_size)
		embed_init = (get_embed, (encoder_name,))
		short_name = 'DP_win{}_numwalks{}_embed{}'.format(win, numWalk, embed_size)
		paras.append((embed_init, short_name))
	return paras


def get_GloveEmbedParas():
	from core.node_embed.GloveEncoder import get_embed
	embed_size_list = [128]
	x_max_list = [10, 100]
	phe_list_modes = [PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP]
	paras = []
	for embed_size, x_max, phe_list_mode in itertools.product(embed_size_list, x_max_list, phe_list_modes):
		encoder_name = 'GloveEncoder_vec{}_xMax{}_max_iter200'.format(embed_size, x_max)
		embed_init = (get_embed, (encoder_name, phe_list_mode))
		short_name = 'GL_embed{}_xMax{}_phe{}'.format(embed_size, x_max, pm_to_short(phe_list_mode))
		paras.append((embed_init, short_name))
	return paras


def get_sdne_embed_paras():
	from core.node_embed.OpenNEEncoder import get_embed
	w_lists = ["[512,128]", "[512,256]"]
	lr_list = [0.001, 0.0001]
	paras = []
	for w, lr in itertools.product(w_lists, lr_list):
		encoder_name = 'encoder{}_lr{}_epoch400_alpha0.000001_beta5_nu1-0.00001_nu2-0.0001'.format(w, lr)
		embed_init = (get_embed, (encoder_name, 'SDNEEncoder'))
		short_name = 'SDNE_encoder{}_lr{}'.format(w, lr)
		paras.append((embed_init, short_name))
	return paras


def get_gcn_embed_paras():
	from core.node_embed.GCNEncoder import get_embed
	a = [
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtW_units64_lr0.01_w_decay5e-06', 'GCN_L_xtW_units64_embed-1', -1, False),
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtW_units128_lr0.01_w_decay5e-06', 'GCN_L_xtW_units128_embed-1', -1, False),
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtW_units128_lr0.01_w_decay5e-06', 'GCN_L_xtW_units128_embed0', 0, False),
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtW_units128_lr0.01_w_decay5e-06', 'GCN_L_xtW_units128_embed1', 1, False),
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtW_units256_lr0.01_w_decay5e-06', 'GCN_L_xtW_units256_embed-1', -1, False),
		('GCNDisAsLabelEncoder', 'DisAsLabel_xtI_units128_lr0.01_w_decay5e-06', 'GCN_L_xtI_units128_embed0', 0, False),
		('GCNDisAsLabelFeatureEncoder', 'DisAsLabelFeature_layer3_units128_lr0.01_w_decay5e-06', 'GCN_LF_layer3_units128_embed0', 0, False),
		('GCNDisAsLabelFeatureEncoder', 'DisAsLabelFeature_layer3_units128_lr0.01_w_decay5e-06', 'GCN_LF_layer3_units128_embed1', 1, False),
		('GCNDisAsLabelFeatureEncoder', 'DisAsLabelFeature_layer3_units128_lr0.01_w_decay5e-06', 'GCN_LF_layer3_units128_embed2', 2, False),
		('GCNDisAsFeatureEncoder', 'DisAsFeature_sigmoid_units128_lr0.0001_w_decay0.0', 'GCN_F_sigm_units128_embed1', 1, True),
		('GCNDisAsFeatureEncoder', 'DisAsFeature_units128_lr0.0001_w_decay0.0', 'GCN_F_units128_embed1', 1, True),
	]
	paras = []
	for encoder_class, encoder_name, short_name, embed_idx, l2_norm in a:
		embed_init = (get_embed, (encoder_name, encoder_class, embed_idx, l2_norm))
		paras.append((embed_init, short_name))
	return paras


def get_hce_embed_paras():
	from core.node_embed.HCEEncoder import get_embed
	a = [
		('HCEEncoder_Adam_bc1_epoch840000_lr0.001_lambda0.0_embedSize64', 'HCE_Adam_lr0.001_lam0.0_embed64'),
		('HCEEncoder_Adam_bc1_epoch840000_lr0.001_lambda0.0_embedSize128', 'HCE_Adam_lr0.001_lam0.0_embed128'),
		('HCEEncoder_Adam_bc1_epoch840000_lr0.001_lambda0.0_embedSize256', 'HCE_Adam_lr0.001_lam0.0_embed256'),
		('HCEEncoder_Adam_bc1_epoch840000_lr0.001_lambda0.1_embedSize128', 'HCE_Adam_lr0.001_lam0.1_embed128'),
		('HCEEncoder_Adam_bc1_epoch840000_lr0.0001_lambda0.0_embedSize128', 'HCE_Adam_lr0.0001_lam0.0_embed128'),
		('HCEEncoder_SGD_bc1_epoch840000_lr0.001_lambda0.0_embedSize128', 'HCE_SGD_lr0.001_lam0.0_embed128')
	]
	paras = []
	for encoder_name, short_name in a:
		embed_init = (get_embed, (encoder_name,))
		paras.append((embed_init, short_name))
	return paras


def cm_to_short(combine_modes):
	d = {VEC_COMBINE_MEAN: 'M', VEC_COMBINE_MAX: 'X'}
	return ''.join([d[m] for m in combine_modes])

def pm_to_short(phe_list_mode):
	d = {PHELIST_ANCESTOR: 'A', PHELIST_REDUCE: 'R', PHELIST_ANCESTOR_DUP: 'D'}
	return d[phe_list_mode]

def process_embed(para):
	c, model_name, embed_init, phe_list_mode, combine_modes = para
	get_embedFunc, get_embed_args = embed_init

	hpo_reader = HPOReader()
	raw_X, y_ = getStandardRawXAndY(hpo_reader, phe_list_mode)
	sw = None

	hpo_embed = get_embedFunc(*get_embed_args)

	model = generate_model(VEC_TYPE_EMBEDDING, phe_list_mode=phe_list_mode, embed_mat=hpo_embed, combine_modes=combine_modes, mode=TRAIN_MODE, model_name=model_name)
	logger = get_logger(model.name)
	model.train(raw_X, y_, sw, c, logger, save_model=False)

	return model

def train_model_with_embed(get_embedParas):
	"""embedding
	"""
	phe_list_mode_list = [PHELIST_REDUCE, PHELIST_ANCESTOR]
	combine_modes_list = [(VEC_COMBINE_MEAN,), (VEC_COMBINE_MAX,)]

	tree_num_list = [100, 500]
	max_leaves_list = [2**5-1, 2**6-1]
	bagging_frac_list = [0.8]
	feature_frac_list = [0.8]

	paras = []
	for embed_init, embed_name in get_embedParas():
		for phe_list_mode, combine_modes, tree_num, max_leaves, bagfrac, feafrac in itertools.product(
				phe_list_mode_list, combine_modes_list, tree_num_list, max_leaves_list, bagging_frac_list, feature_frac_list):
			gbdt_config = GBDTConfig()
			gbdt_config.tree_num = tree_num
			gbdt_config.max_leaves = max_leaves
			gbdt_config.bagging_frac = bagfrac
			gbdt_config.feature_frac = feafrac
			model_name = 'GBDTModel_phe{}_comb{}_tn{}_leaf{}_bag{}_fea{}_{}'.format(
				pm_to_short(phe_list_mode), cm_to_short(combine_modes), tree_num, max_leaves, bagfrac, feafrac, embed_name)
			paras.append((gbdt_config, model_name, embed_init, phe_list_mode, combine_modes))

	mt = ModelTestor()
	mt.load_test_data()
	metric_set = {'RankMedian', 'TopAcc.1', 'TopAcc.10'}

	for para in paras:
		model = process_embed(para)
		mt.cal_metric_and_save(model, data_names=mt.data_names, metric_set=metric_set, use_query_many=True)



if __name__ == '__main__':

	train_model_with_embed(get_deepwalk_embed_paras)







