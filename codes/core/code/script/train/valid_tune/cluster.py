"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from tqdm import tqdm
from copy import deepcopy
import numpy as np
from multiprocessing import Pool

from core.reader.hpo_reader import HPOReader
from core.predict.ml_model import LogisticModel, LogisticConfig, LSVMModel, LSVMConfig
from core.predict.cluster import APCluster, APClusterConfig
from core.predict.cluster import AggCluster, AggClusterConfig
from core.predict.cluster import DbscanCluster, DbscanClusterConfig
from core.predict.cluster import SpeCluster, SpeClusterConfig
from core.predict.cluster import KMedoidCluster, KMedoidClusterConfig
from core.predict.cluster import ClusterClassifyModel
from core.utils.constant import DIS_SIM_MICA, DIS_SIM_JACCARD, DIS_SIM_COSINE, DIS_SIM_EUCLIDEAN, TRAIN_MODE, PREDICT_MODE
from core.utils.constant import VEC_TYPE_0_1, VEC_TYPE_TF_IDF, VEC_TYPE_TF, VEC_TYPE_IDF, VEC_TYPE_EMBEDDING
from core.utils.constant import CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN, CLUSTER_PREDICT_CENTER
from core.script.train.valid_tune.tune import tune, multi_tune, get_default_phe_list_mode, get_embed_mat, flatten_grid, train_best_model
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper


def get_lr_Grid():

	grid = {
		'C': list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)),
		'fit_intercept':[True],
		'vec_type':[VEC_TYPE_0_1],
	}
	return grid


def get_lr_things(d):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)
	kwargs = {'vec_type': vec_type, 'phe_list_mode': phe_list_mode}
	c = LogisticConfig(d)
	generator = ClusterClassifyModel
	return generator, kwargs, c


def get_svm_Grid():
	grid = {
		'C':list(np.linspace(0.00001, 0.0001, 10)) + list(np.linspace(0.0001, 0.001, 10)) + list(np.linspace(0.001, 0.01, 10)),
		'vec_type':[VEC_TYPE_0_1],
	}
	return grid


def get_svm_things(d):
	d = deepcopy(d)
	vec_type = d['vec_type']; del d['vec_type']
	if 'phe_list_mode' in d:
		phe_list_mode = d['phe_list_mode']; del d['phe_list_mode']
	else:
		phe_list_mode = get_default_phe_list_mode(vec_type)
	kwargs = {'vec_type': vec_type, 'phe_list_mode': phe_list_mode}
	c = LSVMConfig(d)
	generator = ClusterClassifyModel
	return generator, kwargs, c


def get_ap_cluster_grid():
	grid = {
		'dis_sim_type': [DIS_SIM_MICA],
		'predict_method': [CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN],
		'topk': [5, 20, 50, 100],
		'damping': list(np.linspace(0.5, 0.95, 10)),
		'preference': list(np.linspace(0.1, 1.0, 10)) + list(np.linspace(1.0, 9.0, 9))
	}

	return grid


def get_ap_things(d):
	d = deepcopy(d)
	c = APClusterConfig(d)
	kwargs = {'c': c, 'mode': TRAIN_MODE}
	return APCluster, kwargs


def get_agg_cluster_grid():
	grid = {
		'dis_sim_type':[DIS_SIM_MICA, DIS_SIM_JACCARD, DIS_SIM_COSINE, DIS_SIM_EUCLIDEAN],
		'predict_method':[CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN],
		'topk':[5, 20, 50, 100],
		'n_clusters': list(range(2, 10, 1)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)),
		'linkage': ['complete', 'average', 'single']
	}
	return grid


def get_agg_things(d):
	d = deepcopy(d)
	c = AggClusterConfig(d)
	kwargs = {'c':c, 'mode':TRAIN_MODE}
	return AggCluster, kwargs


def get_dbs_cluster_grid():
	grid = {
		'dis_sim_type':[DIS_SIM_MICA, DIS_SIM_JACCARD, DIS_SIM_COSINE],
		'predict_method':[CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN],
		'topk':[5, 20, 50, 100],
		'eps': list(np.linspace(0.1, 1.0, 20)),
		'min_samples': list(range(10, 100, 3)),
	}
	return grid


def get_dbs_things(d):
	d = deepcopy(d)
	c = DbscanClusterConfig(d)
	kwargs = {'c':c, 'mode':TRAIN_MODE}
	return DbscanCluster, kwargs


def get_spe_cluster_grid():
	grid = {
		'dis_sim_type':[DIS_SIM_MICA, DIS_SIM_JACCARD, DIS_SIM_COSINE, DIS_SIM_EUCLIDEAN],
		'predict_method': [CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN],
		'topk':[5, 20, 50, 100],
		'affinity': ['precomputed'],
		'n_clusters': list(range(2, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)),
	}

	return grid


def get_spe_things(d):
	d = deepcopy(d)
	c = SpeClusterConfig(d)
	kwargs = {'c':c, 'mode':TRAIN_MODE}
	return SpeCluster, kwargs


def get_kmedoid_cluster_grid():

	grid = {
		'dis_sim_type':[DIS_SIM_MICA],
		'predict_method':[CLUSTER_PREDICT_MEAN_MAX_TOPK],
		'topk':[10, 20, 30, 50, 100, 200],
		'n_cluster':[4, 5, 6, 7, 8],
		'n_init': list(range(10)),
	}

	return grid


def get_kmedoid_things(d):
	d = deepcopy(d)
	c = KMedoidClusterConfig(d)
	kwargs = {'c':c}
	return KMedoidCluster, kwargs


# ==============================================================================================
def get_clt_things(clt_name, d):
	"""
	Returns:
		object: cluster generator
		dict: kwargs of generator
	"""
	if clt_name == APCluster.__name__:
		return get_ap_things(d)
	if clt_name == AggCluster.__name__:
		return get_agg_things(d)
	if clt_name == DbscanCluster.__name__:
		return get_dbs_things(d)
	if clt_name == SpeCluster.__name__:
		return get_spe_things(d)
	if clt_name == KMedoidCluster.__name__:
		return get_kmedoid_things(d)
	assert False


def get_clf_things(clf_name, d):
	"""
	Returns:
		object: model generator
		dict: kwargs of generator
		Config: Config for training
	"""
	if clf_name == LogisticModel.__name__:
		return get_lr_things(d)
	if clf_name == LSVMModel.__name__:
		return get_svm_things(d)
	assert False


def filter_cluster_para_dict(d):
	if (d['predict_method'] == CLUSTER_PREDICT_MEAN or d['predict_method'] == CLUSTER_PREDICT_CENTER) and 'topk' in d:
		del d['topk']
	return d


def get_model(d, model_name, mode, fm_save_clt):
	clt_generator, clt_kwargs = get_clt_things(*d['clt']); clt_kwargs['mode'] = mode
	clf_generator, clf_kwargs, clf_config = get_clf_things(*d['clf']); clf_kwargs['mode'] = mode
	model = ClusterClassifyModel(HPOReader(), clt_generator, clf_generator,
		clt_kwargs=clt_kwargs, clf_kwargs=clf_kwargs, model_name=model_name, mode=mode, fm_save_clt=fm_save_clt)
	return model, clf_config


def get_predict_model(d, model_name):
	model, _ = get_model(d, model_name, mode=PREDICT_MODE, fm_save_clt=False)
	return model


def train_clt_clf_model(d, save_model=False, model_name=None, train_clf_cpu=4, fm_save_clt=False, **kwargs):
	model, clf_config = get_model(d, model_name, TRAIN_MODE, fm_save_clt=fm_save_clt)
	model.train(clf_config, save_model=save_model, train_clf_cpu=train_clf_cpu)
	# model.load()
	return model


def tune_clt_clf_script(clusterName, grid):
	grid = {
		'clt': [(clusterName, filter_cluster_para_dict(d)) for d in flatten_grid(grid)],
		'clf': [
			(LogisticModel.__name__, d) for d in flatten_grid(get_lr_Grid())
		] + [
			(LSVMModel.__name__, d) for d in flatten_grid(get_svm_Grid())
		],
	}

	multi_tune(grid, train_clt_clf_model, ClusterClassifyModel.__name__+'-'+clusterName, search_type='random', max_iter=300, cpu_use=6, test_cpu_use=1)
	hyper_helper = HyperTuneHelper(ClusterClassifyModel.__name__, 'a')
	hyper_helper.draw_score_with_iteration()


def train_cluster_multi_wrap(paras):
	clusterName, d = paras
	print(clusterName)
	clt_generator, clt_kwargs = get_clt_things(clusterName, d)
	clt = clt_generator(**clt_kwargs)
	if not clt.exists():
		clt.train(save_model=True)


def train_cluster_script(generator, grid, cpu_use=12):
	para_list = [(generator.__name__, d) for d in flatten_grid(grid)]
	if cpu_use > 1:
		with Pool(cpu_use) as pool:
			for _ in tqdm(pool.imap_unordered(train_cluster_multi_wrap, para_list), total=len(para_list), leave=False):
				pass
	else:
		for para in tqdm(para_list):
			train_cluster_multi_wrap(para)

# =============================================================================================
def train_best_kmedoid1():
	# [0.241, 0.365, 0.082]
	d = {
		"clt": [
			"KMedoidCluster",
			{
				"dis_sim_type": "DIS_SIM_COSINE",
				"predict_method": "CLUSTER_PREDICT_MEAN_MAX_TOPK",
				"topk": 200,
				"n_cluster": 2,
			}
		],
		"clf": [
			"LSVMModel",
			{
				"C": 1e-05,
				"vec_type": "VEC_TYPE_0_1"
			}
		]
	}
	train_best_model(
		d, train_clt_clf_model, get_predict_model, 'KMedoid-Cosine-LSVM-clt2',
		repeat=1, cpu_use=1, test_cpu_use=1,
	)


def train_best_kmedoid2():
	# [0.20, 0.30, 0.077]
	d = {
		"clt": [
		"KMedoidCluster",
			{
				"dis_sim_type": "DIS_SIM_COSINE",
				"predict_method": "CLUSTER_PREDICT_MEAN_MAX_TOPK",
				"topk": 100,
				"n_cluster": 4,
			}
		],
		"clf": [
		"LSVMModel",
			{
				"C": 5e-05,
				"vec_type": "VEC_TYPE_0_1"
			}
		]
	}
	train_best_model(
		d, train_clt_clf_model, get_predict_model, 'KMedoid-Cosine-LSVM-clt4',
		repeat=5, cpu_use=1, test_cpu_use=1,
	)


def train_best_kmedoid3():
	# [0.218, 0.310, 0.084]
	d = {
		"clt":[
			"KMedoidCluster",
			{
				"dis_sim_type":"DIS_SIM_MICA",
				"predict_method":"CLUSTER_PREDICT_MEAN_MAX_TOPK",
				"topk":100,
				"n_cluster":7,
			}
		],
		"clf":[
			"LogisticModel",
			{
				"C":0.0001,
				"fit_intercept":True,
				"vec_type":"VEC_TYPE_0_1"
			}
		]
	}
	train_best_model(
		d, train_clt_clf_model, get_predict_model, 'KMedoid-MICA-LR-clt7',
		repeat=5, cpu_use=1, test_cpu_use=1,
	)

def train_best_kmedoid4():
	# [0.229, 0.324, 0.087]
	d = {
		"clt": [
			"KMedoidCluster",
			{
				"dis_sim_type": "DIS_SIM_MICA",
				"predict_method": "CLUSTER_PREDICT_MEAN_MAX_TOPK",
				"n_cluster": 5
			}
		],
		"clf": [
		"LogisticModel",
			{
				"C": 1e-05,
				"fit_intercept": True,
				"vec_type": "VEC_TYPE_0_1"
			}
		]
	}
	train_best_model(
		d, train_clt_clf_model, get_predict_model, 'KMedoid-MICA-LR-clt5',
		repeat=1, cpu_use=1, test_cpu_use=1,
	)


if __name__ == '__main__':
	pass
