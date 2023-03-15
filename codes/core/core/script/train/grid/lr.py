"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""
from core.predict.model_testor import ModelTestor
from core.predict.ml_model.LogisticModel import generate_model, LogisticConfig
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, DATA_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING, LOG_PATH, TRAIN_MODE, VEC_TYPE_0_1_DIM_REDUCT
from core.utils.constant import PHELIST_ANCESTOR, RESULT_PATH, PHELIST_REDUCE, VEC_COMBINE_MEAN, VEC_COMBINE_MAX, PHELIST_ANCESTOR_DUP
from core.utils.utils import get_logger
from core.script.showFeature import show_feature_weight
from core.helper.data.data_helper import DataHelper

import os
import joblib
from tqdm import tqdm
from multiprocessing import Pool
import itertools


def train_script1():
	from core.utils.utils import read_train_from_files

	hpo_reader = HPOReader()
	vec_type = VEC_TYPE_0_1  # VEC_TYPE_EMBEDDING
	model = generate_model(vec_type, hpo_reader=hpo_reader, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, embed_path=MODEL_PATH+'/DeepwalkModel/numwalks100/EMBEDDING')

	TRAIN_FOLDER = DATA_PATH + '/preprocess/AnnoDataSet10'
	# raw_X, y_ = read_train(DATA_PATH + '/preprocess/AnnoDataSet/true.txt')
	raw_X, y_, sw = read_train_from_files([
		TRAIN_FOLDER + '/true.txt',
		TRAIN_FOLDER + '/reduce.txt',
		TRAIN_FOLDER + '/rise-lower.txt',
		TRAIN_FOLDER + '/noise.txt',
	], file_weights=None, fix=False)

	lr_config = LogisticConfig()
	logger = get_logger(model.name, log_path=LOG_PATH+os.sep+model.name)
	model.train(raw_X, y_, sw, lr_config, logger)

	test_script(model, logger)


def process(para):
	lr_config, model_name = para

	dh = DataHelper()
	raw_X, y_ = dh.get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	model = generate_model(VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, model_name=model_name)
	logger = get_logger(model.name, log_path=LOG_PATH+os.sep+model.name)
	model.train(raw_X, y_, sw, lr_config, logger)

	hpo_reader = HPOReader()
	folder = RESULT_PATH+os.sep+'FeatureWeight'+os.sep+'LogisticRegression'; os.makedirs(folder, exist_ok=True)
	X = dh.get_train_X(PHELIST_ANCESTOR, VEC_TYPE_0_1)
	row_names, col_names = hpo_reader.get_dis_list(), hpo_reader.get_hpo_list()
	show_feature_weight(model.clf.coef_, model.clf.intercept_, row_names, col_names, X, folder+os.sep+model.name)

	return model


def train_script2():
	CList = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]
	paras = []
	for C in CList:
		lr_config = LogisticConfig()
		lr_config.C = C
		lr_config.fit_intercept = False
		model_name = 'LogisticModel_01_Ances_Bias0_C{}'.format(C)   # 'LogisticModel_01_Ances_l1_C{}'
		paras.append((lr_config, model_name))
	with Pool(8) as pool:   # l1: 24; multiNomial: 8
		for model in tqdm(pool.imap_unordered(process, paras), total=len(paras), leave=False):
			pass


def train_script3():
	"""multinomial
	"""
	solver_list = ['newton-cg', 'sag']  # 'lbfgs is killed due to too much memory occupied
	CList = [0.001, 0.01, 0.1, 1.0, 10]
	paras = []
	for solver, C in itertools.product(solver_list, CList):
		lr_config = LogisticConfig()
		lr_config.multi_class = 'multinomial'
		lr_config.solver = solver
		lr_config.C = C
		lr_config.fit_intercept = False
		model_name = 'LogisticModel_01_Ances_MultiNomial_Bias0_Solver{solver}_C{C}'.format(solver=solver, C=C)
		paras.append((lr_config, model_name))

	with Pool(8) as pool:   # l1: 24; multiNomial: 8
		for model in tqdm(pool.imap_unordered(process, paras), total=len(paras), leave=False):
			pass


def process4(para):
	"""dim reduct
	"""
	lr_config, model_name, rdt_paras = para
	rdtInitializer, rdtArgs, rdt_kargs = rdt_paras

	raw_X, y_ = DataHelper().get_train_raw_Xy(PHELIST_ANCESTOR)
	sw = None

	dim_reductor = rdtInitializer(*rdtArgs, **rdt_kargs)
	dim_reductor.load()

	model = generate_model(VEC_TYPE_0_1_DIM_REDUCT, phe_list_mode=PHELIST_ANCESTOR, mode=TRAIN_MODE, dim_reductor=dim_reductor, model_name=model_name)
	logger = get_logger(model.name, log_path=LOG_PATH+os.sep+model.name)
	model.train(raw_X, y_, sw, lr_config, logger)

	return model


def train_script4():
	"""pca
	"""
	from core.predict.model_testor import ModelTestor
	import train.grid.pca as trainPCA
	from feature.pca import PCADimReductor
	trainPCA.train_script()

	CList = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
	compo_num_list = [0.95, 5000, 8000, 10000] # [0.3, 0.5, 0.7, 0.8, 0.9]
	paras = []
	for compoNum, C in itertools.product(compo_num_list, CList):
		lr_config = LogisticConfig()
		lr_config.C = C
		model_name = 'LogisticModel_01_AncesPCA_Compo{}_C{}'.format(compoNum, C)
		rdt_paras = (PCADimReductor, ('PCADimReductor_Compo{}'.format(compoNum), ), {})
		paras.append((lr_config, model_name, rdt_paras))

	mt = ModelTestor()
	mt.load_test_data()
	with Pool() as pool:
		for model in tqdm(pool.imap_unordered(process4, paras), total=len(paras), leave=False):
			for data_name in mt.data_names:
				mt.call_all_metric_and_save(model, data_name, cpu_use=os.cpu_count(), use_query_many=True)


def print_feature(model_name):
	hpo_reader = HPOReader()
	clf = joblib.load(MODEL_PATH+os.sep+'LogisticModel'+os.sep+model_name+'.m')
	save_folder = RESULT_PATH+os.sep+'FeatureWeight'+os.sep+'LogisticRegression'; os.makedirs(save_folder, exist_ok=True)
	X = DataHelper().get_train_X(PHELIST_ANCESTOR, VEC_TYPE_0_1)
	row_names, col_names = hpo_reader.get_dis_list(), hpo_reader.get_hpo_list()
	show_feature_weight(clf.coef_, clf.intercept_, row_names, col_names, X, save_folder+os.sep+model_name)


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

	raw_X, y_ = DataHelper().get_train_raw_Xy(phe_list_mode)
	sw = None

	hpo_embed = get_embedFunc(*get_embed_args)
	model = generate_model(VEC_TYPE_EMBEDDING, phe_list_mode=phe_list_mode, embed_mat=hpo_embed, combine_modes=combine_modes, mode=TRAIN_MODE, model_name=model_name)
	logger = get_logger(model.name)
	model.train(raw_X, y_, sw, c, logger)

	return model


def train_model_with_embed(get_embedParas):
	"""embedding
	"""
	phe_list_mode_list = [PHELIST_REDUCE, PHELIST_ANCESTOR]
	combine_modes_list = [(VEC_COMBINE_MEAN,), (VEC_COMBINE_MAX,)]
	CList = [0.0001, 0.01, 1.0]
	paras = []
	for embed_init, embed_name in get_embedParas():
		for phe_list_mode, combine_modes, C in itertools.product(phe_list_mode_list, combine_modes_list, CList):
			lr_config = LogisticConfig()
			lr_config.C = C
			model_name = 'LogisticModel_phe{}_comb{}_C{}_{}'.format(pm_to_short(phe_list_mode), cm_to_short(combine_modes), C, embed_name)
			paras.append((lr_config, model_name, embed_init, phe_list_mode, combine_modes))

	mt = ModelTestor()
	mt.load_test_data()
	metric_set = {'RankMedian', 'TopAcc.1', 'TopAcc.10'}


	with Pool(12) as pool:   # l1: 24; multiNomial: 8
		for model in tqdm(pool.imap_unordered(process_embed, paras), total=len(paras), leave=False):
			mt.cal_metric_and_save(model, data_names=mt.data_names, metric_set=metric_set, use_query_many=True)
			model.delete()


def test_script(model, logger):

	logger.info(model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463']))



if __name__ == '__main__':

	train_model_with_embed(get_gcn_embed_paras)


