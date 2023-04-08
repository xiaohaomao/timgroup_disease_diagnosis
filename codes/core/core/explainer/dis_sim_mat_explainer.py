

import os
import numpy as np
import json

from core.explainer.explainer import Explainer
from core.draw.simpledraw import simple_dist_plot
from core.reader.hpo_reader import HPOReader
from core.utils.utils import n_largest_indices, n_smallest_indices
from core.utils.constant import RESULT_PATH

class DisSimMatExplainer(Explainer):
	def __init__(self, m, sim_type='SIM_TYPE', topk=3000):
		"""
		Args:
			m (np.ndarray)
		"""
		super(DisSimMatExplainer, self).__init__()
		assert m.shape[0] == m.shape[1] and (m == m.T).all()
		self.m = m
		self.sim_type = sim_type
		self.SAVE_FOLDER = RESULT_PATH + '/dis_sim'; os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.topk = topk


	def up_pos(self, pos):
		if pos[0] > pos[1]:
			return (pos[1], pos[0])
		return pos


	def get_topk_score_list(self, largest=1):
		"""
		Args:
			largest (float): 1.0 for topk largest; -1.0 for topk smallest
		Returns:
			[(dis_int1, dis_int2, score), ...]
		"""
		m = self.m
		f = n_largest_indices if largest == 1 else n_smallest_indices
		dis_list = self.hpo_reader.get_dis_list()
		diag = m[np.diag_indices(m.shape[0])]
		m[np.diag_indices(m.shape[0])] = -largest * np.inf
		in_set = set()
		ret = []
		for dis_int1, dis_int2 in zip(*f(self.m, self.topk*2)):
			pos = self.up_pos((dis_int1, dis_int2))
			if pos in in_set:
				continue
			in_set.add(pos)
			ret.append([dis_list[dis_int1], dis_list[dis_int2], float(m[dis_int1, dis_int2])])
		m[np.diag_indices(m.shape[0])] = diag
		return ret


	def explain(self):
		"""
		Returns:
			dict: {
				'QUARTILE': [q0, q25, q50, q75, q100],
				'topk_LARGEST': [],
				'topk_SMALLEST': []
			}
		"""
		m = self.m
		d = {
			'QUARTILE': np.percentile(m, [0, 25, 50, 75, 100]).tolist(),
			'{}_LARGEST'.format(self.topk): self.get_topk_score_list(largest=1),
			'{}_SMALLEST'.format(self.topk): self.get_topk_score_list(largest=-1)
		}
		d = self.add_dis_cns_info(d)
		return d


	def explain_and_save(self, path=None):
		d = self.explain()
		path = self.SAVE_FOLDER + '/{}.json'.format(self.sim_type) if path is None else path
		json.dump(d, open(path, 'w'), indent=2, ensure_ascii=False)


	def draw_dist(self, figpath=None, x_lim=(None, None)):
		figpath = self.SAVE_FOLDER + '/{}.png'.format(self.sim_type) if figpath is None else figpath
		simple_dist_plot(figpath, self.m.flatten(), bins=100, x_label='Matrix Value', title='Matrix Value Distribution', figsize=(20, 10), x_lim=x_lim)



