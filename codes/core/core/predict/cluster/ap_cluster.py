import numpy as np
from sklearn.cluster import AffinityPropagation
import os

from core.predict.cluster.cluster import Cluster, ClusterConfig
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.utils.utils import timer
from core.explainer.cluster_explainer import ClusterExplainer


class APClusterConfig(ClusterConfig):
	def __init__(self, d=None):
		super(APClusterConfig, self).__init__()
		self.max_iter = 500
		self.damping = 0.5
		self.preference = None
		if d is not None:
			self.assign(d)


class APCluster(Cluster):
	def __init__(self, c=APClusterConfig(), hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		super(APCluster, self).__init__(c, hpo_reader=hpo_reader, mode=mode, save_folder=save_folder)
		self.name = 'APCluster'


	def get_default_fm_folder(self):
		return MODEL_PATH + '/Cluster/AffinityPropagation'


	@timer
	def train(self, save_model=True):
		print(self.c)
		c = self.c
		sim_mat = DisSimCalculator(self.hpo_reader).get_dis_sim_mat(c.dis_sim_type, **c.sim_kwargs)
		clt = AffinityPropagation(damping=c.damping, preference=c.preference, max_iter=c.max_iter, affinity='precomputed', verbose=True)
		clt.fit(sim_mat)
		self.get_labels(clt)
		print(ClusterExplainer(self).explain())
		if save_model:
			self.save()


if __name__ == '__main__':
	cluster = APCluster(APClusterConfig(), mode=TRAIN_MODE)
	cluster.train()

