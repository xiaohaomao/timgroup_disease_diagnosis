

from copy import deepcopy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os

from core.predict.cluster.cluster import Cluster, ClusterConfig
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.utils.utils import timer
from core.explainer.cluster_explainer import ClusterExplainer


class AggClusterConfig(ClusterConfig):
	def __init__(self, d=None):
		super(AggClusterConfig, self).__init__()
		self.n_clusters = 10
		self.linkage = 'complete'
		if d is not None:
			self.assign(d)


class AggCluster(Cluster):
	def __init__(self, c=AggClusterConfig(), hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		super(AggCluster, self).__init__(c, hpo_reader=hpo_reader, mode=mode, save_folder=save_folder)
		self.name = 'AggCluster'


	def get_default_fm_folder(self):
		return MODEL_PATH + '/Cluster/AgglomerativeClustering'


	def config_to_id_dict(self):
		d = super(AggCluster, self).config_to_id_dict()
		del d['n_clusters']
		return d




	@timer
	def train(self, save_model=True):
		if save_model:
			self.init_save_path(create=True)
		print(self.c)
		c = self.c
		sim_mat = DisSimCalculator(self.hpo_reader).get_dis_sim_mat(c.dis_sim_type, **c.sim_kwargs)
		clt = AgglomerativeClustering(
			n_clusters=c.n_clusters, linkage=c.linkage, affinity='precomputed',

		)
		clt.fit(sim_mat)
		self.get_labels(clt)
		print(ClusterExplainer(self).explain())
		if save_model:
			self.save()


if __name__ == '__main__':
	cluster = AggCluster(AggClusterConfig(), mode=TRAIN_MODE)
	cluster.train()

