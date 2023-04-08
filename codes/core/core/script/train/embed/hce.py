from core.node_embed.encoder import Encoder
from core.node_embed.HCEEncoder import HCEEncoder, HCEConfig, BatchController1, BatchController2
from core.reader.hpo_reader import HPOReader
from core.utils.constant import OPTIMIZER_SGD, OPTIMIZER_ADAM, EMBEDDING_PATH, PHELIST_ANCESTOR
from core.utils.utils import get_file_list
from core.explainer.hpo_embedExplainer import hpo_embedExplainer

from multiprocessing import Pool
import itertools
import os
import numpy as np
from tqdm import tqdm

def process(para):
	c, encoder_name, bc_initializer = para
	hpo_reader = HPOReader()
	bc = bc_initializer(hpo_reader)
	c.vocab_size = hpo_reader.get_hpo_num()
	encoder = HCEEncoder(encoder_name=encoder_name)
	encoder.train(c, bc)


def train_script():
	bc_list = ['bc1']
	epoch_num_list = [840000]
	lr_list = [0.001, 0.0001]
	lambda_list = [0.0, 0.001, 0.01, 0.1]
	embed_size_list = [64, 128, 256, 512]

	paras = []
	for bc_type, epoch_num, lr, lambda_, embed_size in itertools.product(bc_list, epoch_num_list, lr_list, lambda_list, embed_size_list):
		bc_initializer = BatchController1 if bc_type == 'bc1' else BatchController2
		c = HCEConfig()
		c.epoch_num = epoch_num
		c.lr = lr
		c.lambda_ = lambda_
		c.embed_size = embed_size
		encoder_name = 'HCEEncoder_Adam_{}_epoch{}_lr{}_lambda{}_embedSize{}'.format(bc_type, epoch_num, lr, lambda_, embed_size)
		paras.append((c, encoder_name, bc_initializer))



	with Pool(16) as pool:
		for model in pool.imap_unordered(process, paras):
			pass


def train_script2():
	bc_list = ['bc1']
	epoch_num_list = [840000]
	lr_list = [0.001, 0.01]
	lambda_list = [0.0, 0.001, 0.01, 0.1]
	embed_size_list = [64, 128, 256, 512]

	paras = []
	for bc_type, epoch_num, lr, lambda_, embed_size in itertools.product(bc_list, epoch_num_list, lr_list, lambda_list, embed_size_list):
		bc_initializer = BatchController1 if bc_type == 'bc1' else BatchController2
		c = HCEConfig()
		c.epoch_num = epoch_num
		c.lr = lr
		c.lambda_ = lambda_
		c.embed_size = embed_size
		c.optimizer = OPTIMIZER_SGD
		encoder_name = 'HCEEncoder_SGD_{}_epoch{}_lr{}_lambda{}_embedSize{}'.format(bc_type, epoch_num, lr, lambda_, embed_size)
		paras.append((c, encoder_name, bc_initializer))



	with Pool(16) as pool:   # l1: 24; multiNomial: 8
		for model in pool.imap_unordered(process, paras):
			pass


def draw_dist(folder):
	npz_files = get_file_list(folder, lambda filepath: filepath.split('.').pop() == 'npz')
	anno_hpos = list(HPOReader().get_anno_hpo_int_set())
	for path in tqdm(npz_files):
		with np.load(path) as data:
			hpo_embed = data['arr_0']
			hpo_embed = hpo_embed[anno_hpos,]
			prefix, postfix = os.path.splitext(path)
			l2figpath = prefix+'_drawl2.jpg'    # about 4000 embedding with same length (because no annotation HPOs)

			Encoder().draw_embed_hist(hpo_embed, l2figpath, l2_norm=True)
			elefigpath = prefix+'_drawele.jpg'
			Encoder().draw_embed_hist(hpo_embed, elefigpath, l2_norm=False)


if __name__ == '__main__':

	draw_dist(EMBEDDING_PATH+os.sep+'HCEEncoder')


