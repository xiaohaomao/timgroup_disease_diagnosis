

import json
import random
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
import os

from core.predict.cluster.cluster import Cluster, ClusterConfig
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.utils.utils import timer, check_return
from core.explainer.cluster_explainer import ClusterExplainer


class KMedoidClusterConfig(ClusterConfig):
	def __init__(self, d=None):
		super(KMedoidClusterConfig, self).__init__()
		self.n_cluster = 10
		if d is not None:
			self.assign(d)


class KMedoidCluster(Cluster):
	def __init__(self, c=KMedoidClusterConfig(), hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		super(KMedoidCluster, self).__init__(c, hpo_reader=hpo_reader, mode=mode, save_folder=save_folder)
		self.name = 'KMedoidCluster'
		self.label_to_center_dis_int = None

	def get_default_fm_folder(self):
		return MODEL_PATH + '/Cluster/kmedoids'


	def mat_to_list(self, m):
		return [list(m[i]) for i in range(m.shape[0])]


	def init_save_path(self, create=False):
		super(KMedoidCluster, self).init_save_path(create=create)
		self.CENTER_SAVE_JSON = self.SAVE_FOLDER + os.sep + 'labelToCenter.json'


	def save(self):
		super(KMedoidCluster, self).save()
		json.dump(self.get_label_to_center(), open(self.CENTER_SAVE_JSON, 'w'))


	def load(self):
		super(KMedoidCluster, self).load()
		self.label_to_center_dis_int = json.load(open(self.CENTER_SAVE_JSON))
		self.label_to_center_dis_int = {int(k): v for k, v in self.label_to_center_dis_int.items()}


	@check_return('labels')
	def get_labels(self, clt=None):
		"""
		Returns:
			np.ndarray: shape=(sample_num,)
		"""
		labels = np.zeros(shape=(self.hpo_reader.get_dis_num(),), dtype=np.int32)
		for lb, dis_int_list in enumerate(clt.get_clusters()):
			labels[dis_int_list] = lb
		return labels


	@check_return('label_to_center_dis_int')
	def get_label_to_center(self, clt=None):
		medoids = clt.get_medoids()
		return {lb: dis_int for lb, dis_int in enumerate(medoids)}


	@timer
	def train(self, save_model=True):
		print(self.c)
		c = self.c
		sim_mat = DisSimCalculator(self.hpo_reader).get_dis_sim_mat(c.dis_sim_type, **c.sim_kwargs)
		initial_medoids = random.sample(range(self.hpo_reader.get_dis_num()), c.n_cluster)
		clt = kmedoids(self.mat_to_list(self.sim_mat_to_dist_mat(sim_mat)), initial_medoids)
		clt.process()
		self.get_labels(clt)
		self.get_label_to_center(clt)
		print(ClusterExplainer(self).explain())
		if save_model:
			self.save()


if __name__ == '__main__':
	cluster = KMedoidCluster(KMedoidClusterConfig(), mode=TRAIN_MODE)
	cluster.train()

