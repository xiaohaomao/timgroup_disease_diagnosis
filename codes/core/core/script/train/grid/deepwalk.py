

from core.node_embed.DeepwalkEncoder import DeepwalkEncoder, DeepwalkConfig
from core.reader.hpo_reader import HPOReader
from core.utils.utils import getStandardRawXAndY
import itertools


def process(para):
	c, encoder_name = para
	hpo_reader = HPOReader()
	encoder = DeepwalkEncoder(encoder_name)
	encoder.train(c, hpo_reader)


def train_script():
	win_list = [1, 2, 4, 6]
	num_walk_list = [30, 100, 200]
	embed_size_list = [64, 128, 256, 512]

	paras = []
	for win, numWalk, embed_size in itertools.product(win_list, num_walk_list, embed_size_list):
		c = DeepwalkConfig()
		c.window_size = win
		c.num_walks = numWalk
		c.embed_size = embed_size
		encoder_name = 'DeepwalkEncoder_win{}_numwalks{}_embed{}'.format(win, numWalk, embed_size)
		paras.append((c, encoder_name))

	for para in paras:
		process(para)


if __name__ == '__main__':
	train_script()









