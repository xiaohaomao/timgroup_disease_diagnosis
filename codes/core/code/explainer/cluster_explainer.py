

import os
import numpy as np

from core.explainer.explainer import Explainer
from core.draw.simpledraw import simple_dist_plot

class ClusterExplainer(Explainer):
	def __init__(self, clt):
		super(ClusterExplainer, self).__init__()
		self.clt = clt


	def explain(self):
		"""
		Returns:
			dict: {
				'CONFIG': dict,
				'CLUSTER_NUM': int,
				'MAX_CLUSTER_SIZE': int,
				'MIN_CLUSTER_SIZE': int,
				'MEDIAN_CLUSTER_SIZE': int,
				'AVERAGE_CLUSTER_SIZE': int,
				'CLUSTER_SIZE_1_NUM': int
			}
		"""
		d = {
			'CONFIG': self.clt.c.__dict__,
			'CLUSTER_NUM': self.clt.get_cluster_num(),
		}
		lb_to_size = self.clt.get_label_to_size()
		size_list = list(lb_to_size.values())
		d['MAX_CLUSTER_SIZE'] = int(np.max(size_list))
		d['MIN_CLUSTER_SIZE'] = int(np.min(size_list))
		d['MEDIAN_CLUSTER_SIZE'] = float(np.median(size_list))
		d['AVERAGE_CLUSTER_SIZE'] = float(np.mean(size_list))
		d['CLUSTER_SIZE_1_NUM'] = sum([1 for lb, size in lb_to_size.items() if size == 1])
		return d


	def draw(self, folder):
		self.draw_cluster_size(folder)


	def draw_cluster_size(self, folder):
		"""x-axis: cluster size; y-axis: cluster number
		"""
		figpath = folder + os.sep + 'cluster_size.png'
		x = list(self.clt.get_label_to_size().values())
		simple_dist_plot(figpath, x, 50, x_label='Cluster Size', title='Cluster Size Distribution')


