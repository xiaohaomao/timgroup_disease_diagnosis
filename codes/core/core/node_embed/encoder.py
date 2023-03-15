
import numpy as np

from core.draw.simpledraw import simple_line_plot, simple_dist_plot

class Encoder(object):
	def __init__(self):
		self.name = 'Encoder'


	def get_embed(self):
		"""
		Returns:
			np.ndarray: shape=[hpo_num, vec_size]
		"""
		raise NotImplementedError


	def save_embed_txt(self, m, path):
		"""
		Args:
			m (np.ndarray): (item_num, embed_size)
		"""
		m = m.tolist()
		with open(path, 'w') as f:
			f.write('\n'.join( [' '.join(map(lambda v: str(v), v_list)) for v_list in m] ))


	def draw_train_loss(self, figpath, epoch_list, loss_list):
		simple_line_plot(figpath, epoch_list, loss_list, x_label='Epoch', y_label='Train Loss', title='Train Loss Plot')


	def draw_embed_hist(self, embed, figpath, bins=100, l2_norm=True):
		"""
		Args:
			embed (np.ndarray): shape=(sample_num, embed_size)
		"""
		if l2_norm:
			x = np.sqrt(np.power(embed, 2).sum(1))
			simple_dist_plot(figpath, x, bins, 'l2_norm', 'Embedding L2Norm Distribution')
		else:
			simple_dist_plot(figpath, embed.flatten(), bins, 'Element', 'Embedding Element Distribution')



