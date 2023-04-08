import json
import numpy as np
from sklearn.cluster import DBSCAN
import os

from core.predict.cluster.cluster import Cluster, ClusterConfig
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.utils.utils import timer, check_return
from core.explainer.cluster_explainer import ClusterExplainer


class DbscanClusterConfig(ClusterConfig):
	def __init__(self, d=None):
		super(DbscanClusterConfig, self).__init__()
		self.eps = 0.5
		self.min_samples = 5
		if d is not None:
			self.assign(d)


class DbscanCluster(Cluster):
	def __init__(self, c=DbscanClusterConfig(), hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		super(DbscanCluster, self).__init__(c, hpo_reader=hpo_reader, mode=mode, save_folder=save_folder)
		self.name = 'DbscanCluster'


	def get_default_fm_folder(self):
		return MODEL_PATH + '/Cluster/DBSCAN'


	@check_return('labels')
	def get_labels(self, clt=None):
		labels = clt.labels_.copy()
		unq_label_set = set(labels)
		if -1 in unq_label_set:
			unq_label_set.remove(-1)
		newlb = len(unq_label_set)
		for i in range(len(labels)):
			if labels[i] == -1:
				assert newlb not in unq_label_set
				labels[i] = newlb
				newlb += 1
		return labels


	@timer
	def train(self, save_model=True):
		print(self.c)
		c = self.c
		sim_mat = DisSimCalculator(self.hpo_reader).get_dis_sim_mat(c.dis_sim_type, **c.sim_kwargs)
		clt = DBSCAN(eps=c.eps, min_samples=c.min_samples, metric='precomputed')
		clt.fit( self.sim_mat_to_dist_mat(sim_mat) )
		self.get_labels(clt)
		print(ClusterExplainer(self).explain())
		if save_model:
			self.save()


if __name__ == '__main__':
	cluster = DbscanCluster(mode=TRAIN_MODE)
	cluster.train()

