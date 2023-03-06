

import numpy as np
from sklearn.cluster import SpectralClustering
import os

from core.predict.cluster.cluster import Cluster, ClusterConfig
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE, DIS_SIM_EUCLIDEAN
from core.utils.utils import timer
from core.explainer.cluster_explainer import ClusterExplainer
from core.helper.data.data_helper import DataHelper


class SpeClusterConfig(ClusterConfig):
	def __init__(self, d=None):
		super(SpeClusterConfig, self).__init__()
		self.n_clusters = 8
		self.affinity = 'precomputed'   # 'precomputed' | 'rbf' | 'nearest_neighbors'
		self.gamma = 1.0
		self.n_neighbors = 10
		if d is not None:
			self.assign(d)


class SpeCluster(Cluster):
	def __init__(self, c=SpeClusterConfig(), hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		super(SpeCluster, self).__init__(c, hpo_reader=hpo_reader, mode=mode, save_folder=save_folder)
		self.name = 'SpeCluster'


	def get_default_fm_folder(self):
		return MODEL_PATH + '/Cluster/SpectralClustering'


	@timer
	def train(self, save_model=True):
		print(self.c)
		c = self.c
		if c.affinity == 'precomputed':
			X = DisSimCalculator(self.hpo_reader).get_dis_sim_mat(c.dis_sim_type, positive=True, **c.sim_kwargs)
		elif c.affinity == 'rbf' or c.affinity == 'nearest_neighbors':
			assert c.dis_sim_type == DIS_SIM_EUCLIDEAN
			X = DataHelper(self.hpo_reader).get_train_X()
		else:
			assert False
		clt = SpectralClustering(
			n_clusters=c.n_clusters, affinity=c.affinity, gamma=c.gamma, n_neighbors=c.n_neighbors, n_jobs=None
		)
		clt.fit(X)
		self.get_labels(clt)
		print(ClusterExplainer(self).explain())
		if save_model:
			self.save()


if __name__ == '__main__':
	c = SpeClusterConfig()
	c.affinity = 'rbf'; c.dis_sim_type = DIS_SIM_EUCLIDEAN
	cluster = SpeCluster(SpeClusterConfig(), mode=TRAIN_MODE)
	cluster.train()

