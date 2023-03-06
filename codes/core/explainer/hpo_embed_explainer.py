

import numpy as np
from sklearn.manifold import TSNE

from core.reader.hpo_reader import HPOReader
from core.draw.simpledraw import simple_dot_plot
from core.explainer.explainer import Explainer
from core.utils.utils import get_all_descendents, item_list_to_rank_list, slice_list_with_keep_set, timer
from core.patient.gu_patient_generator import DAIG_MAP_DICT


class hpo_embedExplainer(Explainer):
	def __init__(self, m, hpo_reader=HPOReader()):
		"""
		Args:
			m (np.ndarray): (hpo_num, embed_size)
		"""
		super(hpo_embedExplainer, self).__init__()
		self.m = m
		self.hpo_reader = hpo_reader

	@timer
	def cal_tsne(self):
		tsne = TSNE()
		tsne_X = tsne.fit_transform(self.m)
		return tsne_X


	def draw_hpo_tree(self, figpath, root_hpos, tsne_X, add_text=False, diff_size=False):
		"""
		Args:
			root_hpos (list):
		"""
		hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		id2lb = {}
		for root in root_hpos:
			tree_hpos = list(get_all_descendents(root, self.hpo_reader.get_slice_hpo_dict()))
			lb = self.add_hpo_info(root)
			id2lb.update({id: lb for id in item_list_to_rank_list(tree_hpos, hpo_map_rank)})
		self.draw_tsne(figpath, tsne_X, id2lb, add_text=add_text)


	def draw_dis_hpo(self, figpath, hpo_lists, lb_list, tsne_X, add_text=False):
		hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		lb_list = self.add_dis_cns_info(lb_list)

		lb_to_size = {lb: 150 for lb in lb_list}
		id2lb = {}
		for lb, hpo_list in zip(lb_list, hpo_lists):
			id2lb.update({id: lb for id in item_list_to_rank_list(hpo_list, hpo_map_rank)})
		self.draw_tsne(figpath, tsne_X, id2lb, lb_to_size=lb_to_size, add_text=add_text)


	def get_hpo_id_to_text(self, color_ids):
		hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		hpo2depth = self.hpo_reader.get_hpo2depth()
		hpo_int_to_depth = {hpo_map_rank[hpo]:depth for hpo, depth in hpo2depth.items()}
		hpo_list = self.hpo_reader.get_hpo_list()
		text_list = self.add_hpo_info([hpo_list[hpo_int] for hpo_int in color_ids])
		pid2text = {pid:p_text + '-{}'.format(hpo_int_to_depth[pid]) for pid, p_text in zip(color_ids, text_list)}
		return pid2text


	def get_dis_id_to_text(self, color_ids):
		dis_list = self.hpo_reader.get_dis_list()
		text_list = self.add_dis_cns_info([dis_list[dis_int] for dis_int in color_ids])
		pid2text = {pid:p_text for pid, p_text in zip(color_ids, text_list)}
		return pid2text


	def draw_tsne(self, figpath, tsne_X, id2label, label_order=None, lb_to_size=None, lb2style=None, add_text=False, embed_type='HPO'):
		"""
		Args:
			tsne_X (np.ndarray): shape=(sample_num, 2)
			id2label (dict)
			label_order (list)
			embed_type (str): 'HPO': 'DISEASE'
		"""
		p_types = ['others'] * tsne_X.shape[0]
		for id, lb in id2label.items():
			p_types[id] = lb
		label_order = ['others'] + ( label_order or list(set(id2label.values())) )
		if lb_to_size is not None:
			lb_to_size['others'] = 20
		if lb2style is not None:
			lb2style['others'] = 'o'

		pid2text = None
		if add_text:
			id_list = list(id2label.keys())
			if embed_type == 'HPO':
				pid2text = self.get_hpo_id_to_text(id_list)
			if embed_type == 'DIS':
				pid2text = self.get_dis_id_to_text(id_list)

		p_type_label = embed_type
		p_size_label = None if lb_to_size is None else p_type_label
		p_style_label = None if lb2style is None else p_type_label

		simple_dot_plot(figpath, tsne_X[:, 0], tsne_X[:, 1], p_types=p_types, p_type_order=label_order, sizes=lb_to_size, markers=lb2style,
			pid2text=pid2text, x_label='x', y_label='y', p_type_label=p_type_label, p_size_label=p_size_label, p_style_label=p_style_label,
			title='TSNE of HPO Embedding', figsize=(20, 20), palette='Set2')


def get_sca_disease(hpo_reader):
	"""
	Returns:
		list: [disCode1, disCode2, ...]
	"""
	return slice_list_with_keep_set(DAIG_MAP_DICT['SCA'].keys(), set(hpo_reader.get_dis_list()))


if __name__ == '__main__':
	pass







