

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import gc

import codecs

import json

from core.predict.sim_model import MICAModel, MICALinModel, MinICModel, MICAJCModel, SimGICModel
from core.predict.sim_model import GDDPFisherModel, GDDPWeightTOModel, RDDModel, RBPModel
from core.predict.sim_model import EuclideanModel, JaccardModel, CosineModel, DistanceModel
from core.predict.sim_model import SimTOModel, ICTODQAcrossModel, SimTODQAcrossModel, SimTOQReduceModel, ICTOQReduceModel
from core.predict.sim_model import SimTOQReduceDominantICModel, ICTODQAcrossModel2, SimTODominantRandomQReduceModel, ICTOModel
from core.predict.prob_model import CNBModel, MNBModel, HPOProbMNBModel
from core.predict.prob_model import BOQAModel
from core.predict.prob_model import TransferProbModel, TransferProbNoisePunishModel
from core.predict.prob_model import BayesNetModel
from core.predict.ml_model import LogisticModel
from core.predict.ml_model import LSVMModel
from core.predict.ml_model import LRNeuronModel
from core.predict.semi import SemiLRModel
from core.predict.cluster import ClusterClassifyModel
from core.predict.cluster import KMedoidCluster, KMedoidClusterConfig
from core.predict.pvalue_model import RawPValueModel, HistPValueModel
from core.predict.ensemble import OrderedMultiModel
from core.predict.ensemble import RandomModel

from core.predict.model_testor import ModelTestor
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE, TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, DIST_MEAN_TURN, DIST_SHORTEST, PHELIST_ANCESTOR_DUP, SET_SIM_ASYMMAX_QD
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF, VEC_TYPE_EMBEDDING, VEC_COMBINE_SUM, VEC_COMBINE_MAX, VEC_TYPE_TF_IDF, RESULT_PATH, SEED
from core.utils.constant import SORT_S_P, SORT_P_S, SORT_P, PVALUE_HIST_SCORE_MODEL, PVALUE_RAW_SCORE_MODEL, TRAIN_MODE, get_tune_data_names
from core.utils.constant import DISORDER_GROUP_LEAF_LEVEL, DISORDER_GROUP_LEVEL, DISORDER_SUBTYPE_LEVEL, DISORDER_LEVEL
from core.utils.utils import timer
from core.reader import HPOReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.script.train.valid_tune.tune import get_embed_mat
from core.utils.utils import get_logger, delete_logger

TEST_SEED = 777
print('TEST_SEED = {}'.format(TEST_SEED))


def get_real_data_names(eval_data):
	return get_tune_data_names(eval_data)


def get_eval_datas():

	return [TEST_DATA]

def get_data_names():
	return [
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
	# 'MUT_24_CASES_OMIM',
	# 'MUT_24_CASES_ORPHA',
	# 'MUT_24_CASES_CCRD',
	# 'MUT_24_CASES_OMIM_ORPHA',
	# 'MUT_24_CASES_CCRD_ORPHA',
	# 'MUT_24_CASES_CCRD_OMIM',
	# 'MUT_24_CASES_CCRD_OMIM_ORPHA',

	# validation subsets of RAMEDIS using different knowledge bases
	# 'validation_subset_RAMDEIS_CCRD',
	# 'validation_subset_RAMDEIS_OMIM',
	# 'validation_subset_RAMDEIS_ORPHA',
	# 'validation_subset_RAMDEIS_CCRD_OMIM',
	# 'validation_subset_RAMDEIS_CCRD_ORPHA',
	# 'validation_subset_RAMDEIS_OMIM_ORPHA',
	# 'validation_subset_RAMDEIS_CCRD_OMIM_ORPHA',

	# multi_country_test using different knowledge bases
	# 'Multi-country-test_CCRD',
	# 'Multi-country-test_OMIM',
	# 'Multi-country-test_ORPHA',
	# 'Multi-country-test_CCRD_OMIM',
	# 'Multi-country-test_CCRD_ORPHA',
	# 'Multi-country-test_OMIM_ORPHA',
	# 'Multi-country-test_CCRD_OMIM_ORPHA',

	# simulated datasets
	# 'SIM_ORIGIN',
	# 'SIM_NOISE',
	# 'SIM_IMPRE',
	# 'SIM_IMPRE_NOISE',
	# 'SIM_NOISE_IMPRE',

	]





def get_metric_names(levels=None):

	metric_names = [
		# 'Mic.Recall.20', 'Mac.Recall.20',
		'Mic.Recall.10', #'Mac.Recall.10',
		'Mic.Recall.3', #'Mac.Recall.3',
		'Mic.Recall.1', #'Mac.Recall.1',
		'Mic.RankMedian'
	]

	if levels is not None:
		mark = '_'.join(sorted(levels))
		metric_names = [f'{metric_name}_{mark}' for metric_name in metric_names]


	return metric_names


def get_hpo_reader(keep_dnames=None, rm_no_use_hpo=False):

	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']

	return HPOIntegratedDatasetReader(keep_dnames=keep_dnames, rm_no_use_hpo=rm_no_use_hpo)


def get_mt_hpo_reader(keep_dnames=None):


	return get_hpo_reader(keep_dnames=keep_dnames)

def get_model_name_mark():
	return None

def get_model_name_with_mark(model_name):
	mark = get_model_name_mark()
	if mark is None or model_name == 'Phenomizer':
		return model_name
	return f'{model_name}-{mark}'

def get_sort_types():
	return [SORT_S_P, SORT_P_S]


def get_mc_times():
	return [20000]


def get_paper_random_baseline():
	return [

		'MICA-QD-Random',
		'BOQAModel-dp1.0-Random',
		'RDDModel-Ances-Random',
		'GDDPFisherModel-MinIC-Random',
		'RBPModel-Random',
		'MinIC-QD-Random',
		'MICALin-QD-Random',
		'MICAJC-QD-Random',
		'SimGICModel-Random',
		'JaccardModel-Random',
		'SimTOModel-Random',
		'CosineModel-Random'
	]


def get_base_line_initial_paras(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames, rm_no_use_hpo=False)
	all_model_names = get_paper_random_baseline()

	source_to_model_name_to_paras = {
		'INTEGRATE_CCRD_OMIM_ORPHA': {
			'GDDPFisherModel-MICA-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MICA', 'gamma': 3.5}), # CCRD_OMIM_ORPHA, Z: 3.4; INTEGRATED-CJFH: 3.5
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MICA-Random', 'hpo_reader':hpo_reader}
			),
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma': 3.5}), # CCRD_OMIM_ORPHA, Z: 3.8; INTEGRATED-CJFH: 2.3
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha': 0.04}), # CCRD_OMIM_ORPHA, Z: 0.009; INTEGRATED-CJFH: 0.01
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_OMIM_ORPHA': {
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':2.6}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.009}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD_OMIM': {
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':2.6}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.006}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD_ORPHA': {
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':3.1}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.03}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_OMIM': {
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':4.2}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.02}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_ORPHA':{
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':2.3}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.04}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD':{
			'GDDPFisherModel-MinIC-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':3.1}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.07}),  # CCRD_OMIM_ORPHA, Z: 0.009
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'CCRD_OMIM_ORPHA': {
			'GDDPFisherModel-MICA-Random':(
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MICA', 'gamma':3.8}),
					# CCRD_OMIM_ORPHA, Z: 3.4
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MICA-Random', 'hpo_reader':hpo_reader}
			),
			'GDDPFisherModel-MinIC-Random': (
				OrderedMultiModel,
				([
					(GDDPFisherModel, (hpo_reader,), {'phe_sim':'PHE_SIM_MINIC', 'gamma':3.9}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'GDDPFisherModel-MinIC-Random', 'hpo_reader':hpo_reader}
			),
			'RBPModel-Random':(
				OrderedMultiModel,
				([
					(RBPModel, (hpo_reader,), {'alpha':0.009}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'RBPModel-Random', 'hpo_reader':hpo_reader}
			),
		},
		'OMIM_ORPHA':{

		},
		'CCRD_OMIM':{

		},
		'CCRD_ORPHA':{

		},
		'OMIM':{

		},
		'ORPHA':{

		},
		'CCRD': {

		},
	}

	model_name_to_paras = {
		'MICA-Random': (
			OrderedMultiModel,
			([
				(MICAModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'MICA-Random', 'hpo_reader': hpo_reader}
		),
		'MICA-QD-Random': (
			OrderedMultiModel,
			([
				(MICAModel, (hpo_reader,), {'model_name': 'MICA-QD', 'set_sim_method': SET_SIM_ASYMMAX_QD}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'MICA-QD-Random', 'hpo_reader': hpo_reader}
		),
		'BOQAModel-dp1.0-Random': (
			OrderedMultiModel,
			([
				(BOQAModel, (hpo_reader,), {'use_freq':True, 'model_name':'BOQAModel-dp1.0'}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name': 'BOQAModel-dp1.0-Random', 'hpo_reader': hpo_reader}
		),
		'BOQAModel-NoFreq-Random': (
			OrderedMultiModel,
			([
				(BOQAModel, (hpo_reader,), {'use_freq':False, 'model_name':'BOQAModel-NoFreq'}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name': 'BOQAModel-NoFreq-Random', 'hpo_reader': hpo_reader}
		),
		'RDDModel-Ances-Random': (
			OrderedMultiModel,
			([
				(RDDModel, (hpo_reader,), {'phe_list_mode': PHELIST_ANCESTOR}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'RDDModel-Ances-Random', 'hpo_reader': hpo_reader}
		),
		'RDDModel-Reduce-Random': (
			OrderedMultiModel,
			([
				(RDDModel, (hpo_reader,), {'phe_list_mode': PHELIST_REDUCE}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'RDDModel-Reduce-Random', 'hpo_reader':hpo_reader}
		),
		'MinIC-Random':(
			OrderedMultiModel,
			([
				(MinICModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'MinIC-Random', 'hpo_reader':hpo_reader}
		),
		'MinIC-QD-Random': (
			OrderedMultiModel,
			([
				(MinICModel, (hpo_reader,), {'model_name':'MinIC-QD', 'set_sim_method':SET_SIM_ASYMMAX_QD}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'MinIC-QD-Random', 'hpo_reader': hpo_reader}
		),
		'MICALin-Random': (
			OrderedMultiModel,
			([
				(MICALinModel, (hpo_reader,), {'model_name': 'MICALin', 'slice_no_anno': True}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'MICALin-Random', 'hpo_reader': hpo_reader}
		),
		'MICALin-QD-Random': (
			OrderedMultiModel,
			([
				(MICALinModel, (hpo_reader,), {'model_name':'MICALin-QD', 'set_sim_method':SET_SIM_ASYMMAX_QD, 'slice_no_anno': True}),
				(RandomModel, (hpo_reader,), {'seed': TEST_SEED})
			],),
			{'model_name':'MICALin-QD-Random', 'hpo_reader': hpo_reader}
		),
		'MICAJC-Random': (
			OrderedMultiModel,
			([
				(MICAJCModel, (hpo_reader,), {'model_name':'MICAJC', 'slice_no_anno':True}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'MICAJC-Random', 'hpo_reader':hpo_reader}
		),
		'MICAJC-QD-Random': (
			OrderedMultiModel,
			([
				(MICAJCModel, (hpo_reader,),
				{'model_name':'MICAJC-QD', 'set_sim_method':SET_SIM_ASYMMAX_QD, 'slice_no_anno':True}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'MICAJC-QD-Random', 'hpo_reader':hpo_reader}
		),
		'SimGICModel-Random': (
			OrderedMultiModel,
			([
				(SimGICModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'SimGICModel-Random', 'hpo_reader':hpo_reader}
		),
		'JaccardModel-Random': (
			OrderedMultiModel,
			([
				(JaccardModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'JaccardModel-Random', 'hpo_reader':hpo_reader}
		),
		'SimTOModel-Random': (
			OrderedMultiModel,
			([
				(SimTOModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'SimTOModel-Random', 'hpo_reader':hpo_reader}
		),
		'CosineModel-Random': (
			OrderedMultiModel,
			([
				(CosineModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'CosineModel-Random', 'hpo_reader':hpo_reader}
		),
	}

	model_name_to_paras.update(source_to_model_name_to_paras.get(hpo_reader.name, {}))
	return [model_name_to_paras[model_name] for model_name in all_model_names]


def get_paper_sim_model_names():
	return [
		'ICTODQAcross-Ave-Random',
		#'ICTODQAcrossModel-Union-Random',

	]


def get_sim_model_initial_paras(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames, rm_no_use_hpo=False)
	all_model_names = get_paper_sim_model_names()

	model_name_to_paras = {
		'ICTODQAcross-Ave-Random': (
			OrderedMultiModel,
			([
				(ICTODQAcrossModel, (hpo_reader,), {'model_name':'ICTODQAcross-Ave', 'sym_mode':'ave'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'ICTODQAcross-Ave-Random', 'hpo_reader':hpo_reader}
		),
		'ICTODQAcrossModel-Union-Random': (
			OrderedMultiModel,
			([
				(ICTODQAcrossModel, (hpo_reader,), {'model_name':'ICTODQAcross-Union', 'sym_mode':'union'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'ICTODQAcrossModel-Union-Random', 'hpo_reader':hpo_reader}
		),
		'ICTOQReduceModel-Random': (
			OrderedMultiModel,
			([
				(ICTOQReduceModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'ICTOQReduceModel-Random', 'hpo_reader': hpo_reader}
		),
		'ICTOModel-Random': (
			OrderedMultiModel,
			([
				(ICTOModel, (hpo_reader,), {'model_name':'ICTOModel'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'ICTOModel-Random', 'hpo_reader':hpo_reader}
		),
		'SimTOQReduceModel-Random': (
			OrderedMultiModel,
			([
				(SimTOQReduceModel, (hpo_reader,), dict()),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'SimTOQReduceModel-Random', 'hpo_reader': hpo_reader}
		),
	}
	return [model_name_to_paras[model_name] for model_name in all_model_names]


def get_paper_prob_model_names():
	return [
		'HPOProbMNB-Random',

	]


def get_prob_model_initial_paras(keep_dnames=None):
	hpo_reader = get_hpo_reader(keep_dnames=keep_dnames, rm_no_use_hpo=False)

	all_model_names = get_paper_prob_model_names()

	source_to_model_name_to_paras = {
		'PHENOMIZERDIS': {
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.4, 'p2':None, 'child_to_parent_prob':'max',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD_OMIM_ORPHA':{
			'HPOProbMNB-Random': (
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.65, 'p2':None, 'child_to_parent_prob':'sum', # INTEGRATED-CJFH: 0.4-max
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
			'TransferProb-Random': (
				OrderedMultiModel,
				([
					(TransferProbNoisePunishModel, (hpo_reader,), {'model_name':'TransferProb', 'default_prob': 0.02}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name': 'TransferProb-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_OMIM_ORPHA':{
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.5, 'p2':None, 'child_to_parent_prob':'max',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD_OMIM':{
			'HPOProbMNB-Random': (
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.5, 'p2':None, 'child_to_parent_prob':'max',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD_ORPHA':{
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.55, 'p2':None, 'child_to_parent_prob':'ind',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_OMIM':{
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.2, 'p2':None, 'child_to_parent_prob':'max',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_ORPHA':{
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.6, 'p2':None, 'child_to_parent_prob':'sum',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},
		'INTEGRATE_CCRD':{
			'HPOProbMNB-Random':(
				OrderedMultiModel,
				([
					(HPOProbMNBModel, (hpo_reader,), {
						'phe_list_mode':PHELIST_REDUCE, 'p1':0.6, 'p2':None, 'child_to_parent_prob':'max',
						'model_name':'HPOProbMNB'}),
					(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
				],),
				{'model_name':'HPOProbMNB-Random', 'hpo_reader':hpo_reader}
			),
		},

		'INTEGRATE_PHENOMIZERDIS': {
			'HPOProbMNB-Random': (
				OrderedMultiModel,
				([
					 (HPOProbMNBModel, (hpo_reader,), {
						 'phe_list_mode': PHELIST_REDUCE, 'p1': 0.4, 'p2': None, 'child_to_parent_prob': 'max',
						 'model_name': 'HPOProbMNB'}),
					 (RandomModel, (hpo_reader,), {'seed': TEST_SEED})
				 ],),
				{'model_name': 'HPOProbMNB-Random', 'hpo_reader': hpo_reader}
			),
		},

		'CCRD_OMIM_ORPHA': {

		},
		'OMIM_ORPHA':{

		},
		'CCRD_OMIM':{

		},
		'CCRD_ORPHA':{

		},
		'OMIM':{

		},
		'ORPHA':{

		},
		'CCRD':{

		},
	}

	return [source_to_model_name_to_paras[hpo_reader.name][model_name] for model_name in all_model_names]


def get_paper_random_spv_clf():
	return [
		'CNB-Random',
		'NN-Mixup-Random-1',

	]

def get_spv_clf_initial_paras():
	hpo_reader = get_hpo_reader(rm_no_use_hpo=True)
	hpo_reader_with_all_hpo = get_hpo_reader(rm_no_use_hpo=False)
	all_model_names = get_paper_random_spv_clf()

	model_name_to_paras = {
		'CNB-Random':(
			OrderedMultiModel,
			([
				(CNBModel, (hpo_reader_with_all_hpo, VEC_TYPE_0_1, PHELIST_ANCESTOR,), {'model_name':'CNB'}),
				(RandomModel, (hpo_reader_with_all_hpo,), {'seed':TEST_SEED})
			],),
			{'model_name':'CNB-Random', 'hpo_reader':hpo_reader_with_all_hpo}
		),
		'NN-Random-1': (
			OrderedMultiModel,
			([
				(LRNeuronModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'NN-1'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'NN-Random-1', 'hpo_reader':hpo_reader}
		),
		'NN-Mixup-Random-1': (
			OrderedMultiModel,
			([
				(LRNeuronModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'NN-Mixup-1'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'NN-Mixup-Random-1', 'hpo_reader':hpo_reader}
		),
		'NN-Pert-Random-1': (
			OrderedMultiModel,
			([
				(LRNeuronModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'NN-Pert-1'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'NN-Pert-Random-1', 'hpo_reader':hpo_reader}
		),
		'LR-Random':(
			OrderedMultiModel,
			([
				(LogisticModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'LR'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'LR-Random', 'hpo_reader':hpo_reader}
		),
		'SVM-Random':(
			OrderedMultiModel,
			([
				(LSVMModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'SVM'}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'SVM-Random', 'hpo_reader':hpo_reader}
		),
		'MNB-Random':(
			OrderedMultiModel,
			([
				(MNBModel, (hpo_reader,),
				{'model_name':'MNB', 'vec_type':VEC_TYPE_TF, 'phe_list_mode':PHELIST_ANCESTOR_DUP}),
				(RandomModel, (hpo_reader,), {'seed':TEST_SEED})
			],),
			{'model_name':'MNB-Random', 'hpo_reader':hpo_reader}
		),
	}
	return [model_name_to_paras[model_name] for model_name in all_model_names]


def get_paper_ensemble_model_names():
	return [

		#'ICTO(A)-HPOProb',

		#'ICTO(A)-HPOProb-CNB',

		'ICTO(A)-HPOProb-CNB-MLP',
	]

def ensemble_name_to_model_names(ensemble_name):
	short_map_long = {
		'ICTO(U)':'ICTODQAcrossModel-Union-Random',
		'ICTO(A)': 'ICTODQAcross-Ave-Random',
		'HPOProb': 'HPOProbMNB-Random',
		'CNB': 'CNB-Random',
		'MLP': 'NN-Mixup-Random-1'
	}
	split_names = ensemble_name.split('-')
	return [short_map_long[short_name] for short_name in split_names]


def get_semi_initial_paras():
	hpo_reader = get_hpo_reader()
	model_initial_paras = [

		(SemiLRModel, (hpo_reader, VEC_TYPE_TF), {'model_name': 'SemiLR', 'phe_list_mode': PHELIST_ANCESTOR_DUP})
	]
	return model_initial_paras


def get_semi_model_names():
	return [
		'MixMN1',
		'MixMN2',

		'SemiLR'
	]


def get_clt_clf_initial_paras():
	hpo_reader = get_hpo_reader()
	model_initial_paras = [

		(ClusterClassifyModel, (hpo_reader,), {
			'clt_generator':KMedoidCluster, 'clt_kwargs':{'c':KMedoidClusterConfig({
				"dis_sim_type": "DIS_SIM_MICA",
				"predict_method": "CLUSTER_PREDICT_MEAN_MAX_TOPK",
				"n_cluster": 5
			})}, 'clf_generator':LogisticModel,
			'clf_kwargs':{"vec_type":"VEC_TYPE_0_1"},
			'model_name':'KMedoid-MICA-LR-clt5'
		}),
	]
	return model_initial_paras


def get_clt_clf_model_names():
	return [
		'KMedoid-Cosine-LSVM-clt2',
		'KMedoid-Cosine-LSVM-clt4',

		'KMedoid-MICA-LR-clt5',
	]


def get_model_initial_paras():
	hpo_reader = get_hpo_reader()
	model_initial_paras = [
		(SimTOModel, (hpo_reader,), dict()),
		(SimTOQReduceModel, (hpo_reader,), dict()),

		(ICTODQAcrossModel, (hpo_reader,), dict()),
		(LogisticModel, (hpo_reader, VEC_TYPE_0_1), {'model_name': 'LogisticModel_01_Ances_Bias0_C0.05'}),
		(LSVMModel, (hpo_reader, VEC_TYPE_0_1), {'model_name':'LSVMModel_01_Ances_C0.001'}),
		(MNBModel, (hpo_reader,), {'model_name':'MNBModel_alpha0.01'}),
		(MNBModel, (hpo_reader,), {'model_name':'MNBModel_alpha0.001'}),
		(CNBModel, (hpo_reader, VEC_TYPE_0_1, PHELIST_ANCESTOR,), {'model_name':'CNBModel_01_alpha500.0'}),

		(MICAModel, (hpo_reader,), dict()),
		(MICALinModel, (hpo_reader,), dict()),
		(GDDPFisherModel, (hpo_reader,), HyperTuneHelper(GDDPFisherModel.__name__).get_best_para()),
		(GDDPWeightTOModel, (hpo_reader,), dict()),
		(EuclideanModel, (hpo_reader,), dict()),
	]
	return model_initial_paras


def get_draw_model_names():
	return [
		'SimTOModel',
		'SimTOQReduceModel',

		'ICTODQAcrossModel',
		'LogisticModel_01_Ances_Bias0_C0.05',
		'LSVMModel_01_Ances_C0.001',
		'MNBModel_alpha0.01',
		'MNBModel_alpha0.001',
		'CNBModel_01_alpha500.0',
		'MICAModel',
		'MICALinModel',
		'GDDPFisherModel',
		'GDDPWeightTOModel',
		'EuclideanModel',
	]


def cal_metric(model_initial_paras, save_raw_results=True, cpu_use=8, use_query_many=True,
		keep_general_dis_map=True, rd_decompose=False):

	logger = get_logger('testOptimalModel')
	model = None


	for initializer, args, kwargs in model_initial_paras:
		del model; gc.collect()
		model = initializer(*args, **kwargs)
		save_model_name = get_model_name_with_mark(model.name)

		for eval_data in get_eval_datas():
			mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)

			mt.load_test_data(get_data_names())



			mt.cal_metric_and_save(
				model, data_names=get_data_names(), metric_set=set(get_metric_names()), cpu_use=cpu_use,
				use_query_many=use_query_many, save_raw_results=save_raw_results, logger=logger, save_model_name=save_model_name,
				rd_decompose=rd_decompose
			)

	delete_logger(logger)


def cal_metric_from_raw(model_names, keep_general_dis_map=True):
	logger = get_logger('testOptimalModel')
	model_names = [get_model_name_with_mark(model_name) for model_name in model_names]



	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		data_names = get_data_names() or mt.data_names


		for model_name, data_name in itertools.product(model_names, data_names):
			mt.cal_and_save_performance_from_raw_results(model_name, data_name, metric_set=set(get_metric_names()), logger=logger)


def cal_pvalue_metric(model_initial_paras, pmodelGenerator, *pmargs, **pmkwargs):
	model = None
	for initializer, args, kwargs in model_initial_paras:
		del model; gc.collect()
		model = initializer(*args, **kwargs)
		pmodel = pmodelGenerator(model, *pmargs, **pmkwargs)

		pmodel.train(12)
		for eval_data in [TEST_DATA]:
			mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED)
			mt.load_test_data()

			mt.cal_pvalue_metric_and_save(pmodel, get_real_data_names(eval_data), metric_set=set(get_metric_names()), sort_types=get_sort_types(), save_raw_results=True)


def get_p_model_names(model_name):
	return [model_name] + [
		'{}-{}-{}-{}-{}'.format(model_name, pType, mc, sort_type, pcorrect)
		for pType, mc, sort_type, pcorrect in itertools.product(['RAW'], get_mc_times(), get_sort_types(), [None, 'fdr_bh'])
	]


def draw_pvalue(model_name):
	for eval_data in [TEST_DATA]:
		mt = ModelTestor(eval_data, hpo_reader=get_hpo_reader(), seed=TEST_SEED)

		mt.draw_metric_bar(get_real_data_names(eval_data), get_metric_names(), get_p_model_names(model_name))


@timer
def draw(model_names):

	for eval_data in [TEST_DATA, VALIDATION_DATA]:
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED)

		mt.draw_metric_bar(mt.data_names, get_metric_names(), model_names)


def gen_excel(model_names, cal_model_rank=True, levels=None, cal_dataset_mean=True, conf_level=None, keep_general_dis_map=True):
	model_names = [get_model_name_with_mark(model_name) for model_name in model_names]


	for eval_data in get_eval_datas():



		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_result_xlsx(get_metric_names(levels), model_names, get_data_names() or mt.data_names,
			cal_model_rank=cal_model_rank, cal_dataset_mean=cal_dataset_mean, conf_level=conf_level)


def gen_ave_source_ranking_excel(model_names, source_lists, keep_general_dis_map=False, use_mark=True):
	prefix = 'INTEGRATE_' if get_hpo_reader().name.startswith('INTEGRATE_') else ''
	source_marks = [prefix+'_'.join(sorted(sources)) for sources in source_lists]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_ave_source_ranking_excel('Mic.RankMedian', model_names, get_data_names() or mt.data_names, source_marks, use_mark=use_mark)


def gen_source_compare_pvalue_excel(model_names, sources_pairs, alternative='two.sided',
		metric='Mic.RankMedian', multi_test_cor=None, keep_general_dis_map=False):
	source_marks_pairs = [('_'.join(sorted(sources1)), '_'.join(sorted(sources2))) for sources1, sources2 in sources_pairs]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_source_compare_pvalue_excel(model_names, get_data_names() or mt.data_names, source_marks_pairs,
			alternative=alternative, metric=metric, multi_test_cor=multi_test_cor)


def gen_source_compare_int_excel(model_names, sources_pairs, metric='Mic.RankMedian',
		conf_level=0.95, cpu_use=12, keep_general_dis_map=False, multi_mean=True, use_mark=True):
	prefix = 'INTEGRATE_' if get_hpo_reader().name.startswith('INTEGRATE_') else ''
	source_marks_pairs = [(prefix+'_'.join(sorted(sources1)), prefix+'_'.join(sorted(sources2))) for sources1, sources2 in sources_pairs]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_source_compare_diff_int_excel(model_names, get_data_names() or mt.data_names, source_marks_pairs,
			metric=metric, conf_level=conf_level, cpu_use=cpu_use, multi_mean=multi_mean, use_mark=use_mark)


def gen_average_kb_ranking_excel(model_names, sources_pairs, metric='Mic.RankMedian', use_mark=True):
	prefix = 'INTEGRATE_' if get_hpo_reader().name.startswith('INTEGRATE_') else ''
	source_marks_pairs = [(prefix + '_'.join(sorted(sources1)), prefix + '_'.join(sorted(sources2))) for
		sources1, sources2 in sources_pairs]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED,
			keep_general_dis_map=keep_general_dis_map)
		mt.gen_average_kb_ranking_excel(model_names, get_data_names() or mt.data_names, source_marks_pairs,
			metric=metric, use_mark=use_mark)
	pass


def gen_pairwise_pvalue_excel(model_names1, model_names2, alternative='two.sided',
		metric='Mic.RankMedian', multi_test_cor=None, keep_general_dis_map=True):
	model_names1 = [get_model_name_with_mark(model_name) for model_name in model_names1]
	model_names2 = [get_model_name_with_mark(model_name) for model_name in model_names2]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_pvalue_table(model_names1, model_names2, data_names=get_data_names() or mt.data_names,
			alternative=alternative, metric=metric, multi_test_cor=multi_test_cor)


def gen_dataset_pavalue_excel(model_names, data_name_pairs, alternative='two.sided',
			metric='Mic.RankMedian', multi_test_cor=None, cpu_use=12, keep_general_dis_map=True):
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_dataset_pavalue_excel(model_names, data_name_pairs,
			alternative=alternative, metric=metric, multi_test_cor=multi_test_cor, cpu_use=cpu_use)


def gen_pairwise_diff_int_excel(model_names1, model_names2, conf_level=0.95,
		metric='Mic.RankMedian', cpu_use=12, keep_general_dis_map=True, sheet_order_by_swap=False):
	model_names1 = [get_model_name_with_mark(model_name) for model_name in model_names1]
	model_names2 = [get_model_name_with_mark(model_name) for model_name in model_names2]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_pairwise_diff_int_excel(model_names1, model_names2, data_names=get_data_names() or mt.data_names,
			conf_level=conf_level, metric=metric, cpu_use=cpu_use, sheet_order_by_swap=sheet_order_by_swap)


def gen_pairwise_multi_mean_diff_int_excel(model_names1, model_names2, conf_level=0.95,
		metric='Mic.RankMedian', cpu_use=12, keep_general_dis_map=True):
	model_names1 = [get_model_name_with_mark(model_name) for model_name in model_names1]
	model_names2 = [get_model_name_with_mark(model_name) for model_name in model_names2]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_pairwise_multi_mean_diff_conf_int_excel(model_names1, model_names2,
			data_names=get_data_names() or mt.data_names, conf_level=conf_level, metric=metric, cpu_use=cpu_use)


def gen_dataset_diff_int_excel(model_names, data_name_pairs, conf_level=0.95,
		metric='Mic.RankMedian', cpu_use=12, keep_general_dis_map=True):
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.gen_dataset_diff_int_excel(model_names, data_name_pairs, conf_level, metric, cpu_use)


def cal_levels_metrics(model_names, levels):
	model_names = [get_model_name_with_mark(model_name) for model_name in model_names]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED)
		for model_name, data_name in itertools.product(model_names, get_data_names() or mt.data_names):
			mt.cal_level_performance(model_name, data_name, levels, get_metric_names(), save=True)


def cal_conf_int(model_names, conf_level=0.95, cal_multi=True, cal_single=True, cpu_use=12, levels=None, keep_general_dis_map=True, rd_decompose=False):
	model_names = [get_model_name_with_mark(model_name) for model_name in model_names]
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		mt.cal_metric_conf_int_and_save_parallel(model_names, data_names=get_data_names() or mt.data_names,
			conf_level=conf_level, metric_set=get_metric_names(levels), cal_multi=cal_multi, cpu_use=cpu_use, cal_single=cal_single)


def process_phenomizer(keep_general_dis_map=False):
	mt = ModelTestor(TEST_DATA, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
	mt.process_phenomizer_results(get_metric_names())


@timer
def draw_hpo_change(model_names, target='patient'):
	mt = ModelTestor(hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED)
	metric_set = {'TopAcc.10', 'TopAcc.1'}
	min_hpo, max_hpo = 1, 41

	for data_name in mt.data_names:
		mt.draw_patient_num_with_hpo_length(data_name, min_hpo=min_hpo, max_hpo=max_hpo, target='disease')


@timer
def gen_dis_category_result(model_names, keep_general_dis_map=True):
	metric_set = {'Mic.Recall.10', 'Mic.Recall.1', 'Mic.RankMedian'}
	hpo_reader = get_mt_hpo_reader()
	for eval_data in get_eval_datas():
		mt = ModelTestor(eval_data, hpo_reader=hpo_reader, seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		data_names = get_data_names() or mt.data_names
		for data_name in data_names:
			print(eval_data, data_name)
			mt.gen_dis_category_result_xlsx(data_name, model_names, metric_set,
				folder=os.path.join(RESULT_PATH, hpo_reader.name, 'DisCategoryResult', eval_data), reverse=False)


def gen_case_result(model_names, keep_general_dis_map=True):
	hpo_reader = get_mt_hpo_reader()


	for eval_data in get_eval_datas():

		mt = ModelTestor(eval_data, hpo_reader=hpo_reader, seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		data_names = get_data_names() or mt.data_names
		for data_name in data_names:

			mt.gen_case_result_xlsx(data_name, model_names, folder=os.path.join(RESULT_PATH, hpo_reader.name, 'CaseResult', eval_data))


def rank_ensemble(ensemble_names , data_names=None, model_weight=None,
		save_raw_results=True, cpu_use=8, combine_method='ave', keep_general_dis_map=True, rd_decompose=False):
	for ensemble_name in ensemble_names:
		model_names = ensemble_name_to_model_names(ensemble_name)
		model_names = [get_model_name_with_mark(model_name) for model_name in model_names]

		ensemble_name = get_model_name_with_mark(ensemble_name)
		logger = get_logger('testOptimalModel')

		for eval_data in get_eval_datas():
			mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
			data_names = data_names or get_data_names() or mt.data_names
			mt.load_test_data(data_names)
			mt.rank_score_ensemble(model_names, data_names, ensemble_name, model_weight,
				metric_set=set(get_metric_names()), logger=logger, save=save_raw_results,
				cpu_use=cpu_use, combine_method=combine_method, hpo_reader=get_hpo_reader())
		delete_logger(logger)


def consistency_ensemble(tgt_model_name, all_model_names, ensemble_name, topk, threshold,
		data_names=None, save_raw_results=True, keep_general_dis_map=True):
	logger = get_logger('testOptimalModel')

	for eval_data in [TEST_DATA]:
		mt = ModelTestor(eval_data, hpo_reader=get_mt_hpo_reader(), seed=TEST_SEED, keep_general_dis_map=keep_general_dis_map)
		data_names = data_names or mt.data_names
		mt.load_test_data(data_names)
		mt.consistency_ensemble(tgt_model_name, all_model_names, data_names, ensemble_name, topk, threshold,
			metric_set=set(get_metric_names()), logger=logger, save=save_raw_results, cpu_use=8, hpo_reader=get_hpo_reader())
	delete_logger(logger)


if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	print('============ the data path ==============',RESULT_PATH)


	raw_data_path = RESULT_PATH




	keep_general_dis_map = True
	conf_level = 0.95 #  0.95 # 0.95
	save_raw_results = True

	# # Calculate metric ===================================
	#cal_metric(get_base_line_initial_paras(), cpu_use=12, save_raw_results=save_raw_results, keep_general_dis_map=keep_general_dis_map)
	cal_metric(get_sim_model_initial_paras(), cpu_use=12, save_raw_results=save_raw_results, keep_general_dis_map=keep_general_dis_map)
	cal_metric(get_prob_model_initial_paras(), cpu_use=12, save_raw_results=save_raw_results, keep_general_dis_map=keep_general_dis_map)
	cal_metric(get_spv_clf_initial_paras(), cpu_use=12, save_raw_results=save_raw_results, keep_general_dis_map=keep_general_dis_map)
	rank_ensemble(get_paper_ensemble_model_names(), cpu_use=8, save_raw_results=save_raw_results, combine_method='ave', keep_general_dis_map=keep_general_dis_map)
	#
	#

	# # General ===================================
	# #our_methods = get_paper_sim_model_names()
	# #our_methods = get_paper_sim_model_names() + get_paper_prob_model_names() + get_paper_random_spv_clf()
	our_methods = get_paper_sim_model_names() + get_paper_prob_model_names() + get_paper_random_spv_clf() + get_paper_ensemble_model_names()
	# #our_methods = get_paper_sim_model_names() + get_paper_prob_model_names()  + get_paper_ensemble_model_names()
	# #our_methods = get_paper_prob_model_names() + get_paper_ensemble_model_names()
	# #our_methods = get_paper_prob_model_names()
	# #our_methods = get_paper_random_spv_clf()
	# #our_methods = get_paper_ensemble_model_names()
	#

	baseline_methods = get_paper_random_baseline()

	#cal_conf_int(baseline_methods+our_methods, cpu_use=12, conf_level=conf_level, cal_single=True, keep_general_dis_map=keep_general_dis_map)
	cal_conf_int(our_methods, cpu_use=12, conf_level=conf_level, cal_single=True, keep_general_dis_map=keep_general_dis_map)
	#

	#cal_metric_from_raw(model_names=baseline_methods+our_methods, keep_general_dis_map=keep_general_dis_map)
	cal_metric_from_raw(model_names=our_methods, keep_general_dis_map=keep_general_dis_map)

	#gen_excel(baseline_methods+our_methods, cal_model_rank=True, cal_dataset_mean=True, conf_level=conf_level, keep_general_dis_map=keep_general_dis_map)
	gen_excel(our_methods, cal_model_rank=True, cal_dataset_mean=True, conf_level=conf_level,keep_general_dis_map=keep_general_dis_map)
	#gen_dis_category_result(baseline_methods+our_methods)
	gen_dis_category_result(our_methods)
	# #gen_dis_category_result(['JaccardModel-Random', 'CosineModel-Random', 'SimGICModel-Random'] + our_methods)
	#
	#
	#gen_case_result(baseline_methods+our_methods)
	gen_case_result(our_methods)

