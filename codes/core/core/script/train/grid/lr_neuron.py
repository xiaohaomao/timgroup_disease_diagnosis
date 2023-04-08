from mlModel.LRNeuronModel import LRNeuronConfig, generate_model
from core.utils.constant import TRAIN_MODE
from core.predict.model_testor import ModelTestor

import itertools
import os

def process(para):
	c, model_name = para
	model = generate_model(mode=TRAIN_MODE, model_name=model_name)
	model.train(c, from_last=False)


def train_dropout():
	"""dropout
	"""
	mt = ModelTestor()
	keep_prob_list = [1.0, 0.5]

	paras = []
	for keep_prob in keep_prob_list:
		c = LRNeuronConfig()
		c.keep_prob = keep_prob
		c.mt_test_data_names = mt.data_names
		model_name = 'LRNeuronModel_keep_prob{}'.format(keep_prob)
		paras.append((c, model_name))

	for para in paras:
		process(para)


def train_wdecay():
	"""w_decay
	"""
	mt = ModelTestor()
	w_decay_list = [1e-6, 1e-5, 1e-4]

	paras = []
	for w_decay in w_decay_list:
		c = LRNeuronConfig()
		c.w_decay = w_decay
		c.mt_test_data_names = mt.data_names
		model_name = 'LRNeuronModel_w_decay{}'.format(w_decay)
		paras.append((c, model_name))

	for para in paras:
		process(para)


def train_mixup():
	"""mixup
	"""
	mt = ModelTestor()
	keep_prob_list = [1.0, 0.5]
	alpha_list = [0.2, 0.8, 1.0, 4.0]

	paras = []
	for keep_prob, alpha in itertools.product(keep_prob_list, alpha_list):
		c = LRNeuronConfig()
		c.mixup = True
		c.mix_alpha = alpha
		c.keep_prob = keep_prob
		c.mt_test_data_names = mt.data_names
		model_name = 'LRNeuronModel_mixup{}_keep_prob{}'.format(alpha, keep_prob)
		paras.append((c, model_name))

	for para in paras:
		process(para)


if __name__ == '__main__':
	pass
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	train_wdecay()




