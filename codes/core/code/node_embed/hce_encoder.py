

from core.node_embed.encoder import Encoder
from core.predict.config import Config
import tensorflow as tf
from core.utils.constant import EMBEDDING_PATH, OPTIMIZER_SGD, OPTIMIZER_ADAM, DATA_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR
from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_logger, get_all_ancestors_with_dist, check_load_save
from core.utils.utils_tf import get_optimizer
from core.draw.simpledraw import simple_line_plot

import os
import random
import numpy as np
from tqdm import tqdm
import heapq
from multiprocessing import Pool


class HCEConfig(Config):
	def __init__(self):
		super(HCEConfig, self).__init__()
		self.batch_size = 128
		self.embed_size = 256
		self.vocab_size = None   # remember to set before training
		self.epoch_num = 540000  # 34739568 / 128 = 271402
		self.lr = 0.001
		self.optimizer = OPTIMIZER_ADAM
		self.lambda_ = 0.0  # l2-regular


class HCEEncoder(Encoder):
	def __init__(self, encoder_name=None):
		super(HCEEncoder, self).__init__()
		self.name = 'HCEEncoder' if encoder_name is None else encoder_name
		folder = EMBEDDING_PATH + os.sep + 'HCEEncoder'; os.makedirs(folder, exist_ok=True)
		self.EMBED_PATH = folder + os.sep + self.name + '.npz'
		self.CONFIG_JSON = folder + os.sep + self.name + '.json'
		self.LOG_PATH = folder + os.sep + self.name + '.log'
		self.LOSS_FIG_PATH = folder + os.sep + self.name + '.jpg'
		self.hpo_embed = None    # np.ndarray; shape=(HPO_CODE_NUM, embed_size)


	def get_embed(self, setzero=False):
		"""
		Returns:
			np.ndarray: shape=[hpo_num, vec_size]
		"""
		if self.hpo_embed is None:
			with np.load(self.EMBED_PATH) as data:
				self.hpo_embed = data['arr_0']
		if setzero:
			hpo_reader = HPOReader()
			no_anno_hpos = list(set(range(hpo_reader.get_hpo_num())) - set(hpo_reader.get_hpo_int_to_dis_int(PHELIST_ANCESTOR).keys()))
			self.hpo_embed[no_anno_hpos,] = 0.0
		return self.hpo_embed


	def build(self, c):
		self.inputs = tf.placeholder(tf.int32, shape=(None,))    # [batch_size]
		self.labels = tf.placeholder(tf.int32, shape=(None,))    # [batch_size]
		self.weights = tf.placeholder(tf.float32, shape=(None,))  # [batch_size]
		self.lr = tf.placeholder(tf.float32, name='lr')

		self.embedding = tf.get_variable(   # [vocab_size, embed_size]
			'embedding',
			shape=[c.vocab_size, c.embed_size],
			dtype=tf.float32,
			initializer=tf.contrib.layers.xavier_initializer(uniform=True)
		)

		embed = tf.nn.embedding_lookup(self.embedding, self.inputs) # [batch_size, embed_size]
		self.logits = tf.matmul(embed, self.embedding, transpose_b=True) # [batch_size, vocab_size]
		labels_one_hot = tf.one_hot(self.labels, c.vocab_size)
		CE = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits) # [batch_size]
		self.loss = tf.reduce_mean(tf.multiply(CE, self.weights))
		self.loss += c.lambda_ * tf.nn.l2_loss(self.embedding)

		optimizer = get_optimizer(c.optimizer, self.lr)
		self.train_op = optimizer.minimize(self.loss)


	def train(self, c, bc):
		logger = get_logger(self.name, log_path=self.LOG_PATH, mode='w')
		logger.info(self.name)
		logger.info(c)

		with tf.variable_scope(self.name):  #
			self.build(c)

		init_op = tf.global_variables_initializer()
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		with tf.Session(config=sess_config) as sess:
			epoch_list, loss_list = [], []
			sess.run(init_op)
			for i in range(1, c.epoch_num+1):
				inputs, labels, weights = bc.next(c.batch_size)
				_, loss = sess.run([self.train_op, self.loss], feed_dict={
					self.inputs: inputs,
					self.labels: labels,
					self.weights: weights,
					self.lr: c.lr
				})
				if i % 500 == 0:
					logger.info('Epoch {}({:.4}%): Batch Loss={}'.format(i, 100*i/c.epoch_num, loss))
					epoch_list.append(i); loss_list.append(loss)
			self.hpo_embed = self.embedding.eval()
			np.savez_compressed(self.EMBED_PATH, self.hpo_embed)
			c.save(self.CONFIG_JSON)
			self.draw_train_loss(self.LOSS_FIG_PATH, epoch_list, loss_list)


class BaseBatchController(object):
	def __init__(self):
		pass

	def gen_data_for_pair(self, et, ec, et_ancestor_dict):
		"""
		Args:
			et (str): hpo_int
			ec (str): hpo_int
			et_ancestor_dict: {hpo_int: distant}
		Returns:
			list: [(et, ec, w), (etAnces, ec, w), ...]
		"""
		et_ancestor_dict = et_ancestor_dict
		w_sum = 0
		for code in et_ancestor_dict:
			w = 1 / (1 + et_ancestor_dict[code])
			et_ancestor_dict[code] = w
			w_sum += w
		ret = [(et, ec, 1.0)] + [(etAnces, ec, w/w_sum) for etAnces, w in et_ancestor_dict.items()]
		return ret


class BatchController1(BaseBatchController):
	def __init__(self, hpo_reader):
		super(BatchController1, self).__init__()
		self.hpo_reader = hpo_reader
		self.data = self.gen_data()  # np.array([(etRank, ec_rank, w), ...])
		self.rank_list = list(range(len(self.data)))
		self.current_rank = 0


	def gen_data(self):
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_REDUCE)
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		ret = []
		for hpoCorpus in tqdm(dis_int_to_hpo_int.values()):
			for et in hpoCorpus:
				et_ancestor_dict = get_all_ancestors_with_dist(et, hpo_int_dict, contain_self=False)
				for ec in hpoCorpus:
					if et == ec:
						continue
					ret.extend(self.gen_data_for_pair(et, ec, et_ancestor_dict))
		print('(et, ec) pair number = {}'.format(len(ret))) # 34739568
		return np.array(ret)


	def next(self, batch_size):
		"""
		Args:
			batch_size (int)
		Returns:
			np.ndarray: inputs, shape=[batch_size], dtype=numpy.int32
			np.ndarray: labels, shape=[batch_size], dtype=numpy.int32
			np.ndarray: weights, shape=[batch_size], dtype=numpy.float32
		"""
		if len(self.data) - self.current_rank < batch_size:
			random.shuffle(self.rank_list)
			self.current_rank = 0
		batch = [self.data[rank] for rank in self.rank_list[self.current_rank: self.current_rank+batch_size]]
		inputs, labels, weights = list(zip(*batch))
		self.current_rank += batch_size
		return np.array(inputs, dtype=np.int32), np.array(labels, dtype=np.int32), np.array(weights, dtype=np.float32)


class BatchController2(BaseBatchController):
	def __init__(self, hpo_reader):
		super(BatchController2, self).__init__()
		self.hpo_reader = hpo_reader
		self.data = self.gen_data()  # np.array([[(etRank, ec_rank, w), ...], ...])
		self.DATA_SIZE = len(self.data)
		self.rank_list = list(range(self.DATA_SIZE))
		self.current_rank = 0


	def gen_data(self):
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int(PHELIST_REDUCE)
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		ret = []
		for hpoCorpus in tqdm(dis_int_to_hpo_int.values()):
			for et in hpoCorpus:
				et_ancestor_dict = get_all_ancestors_with_dist(et, hpo_int_dict, contain_self=False)
				for ec in hpoCorpus:
					if et == ec:
						continue
					ret.append(self.gen_data_for_pair(et, ec, et_ancestor_dict)) # different from BC1
		return np.array(ret)


	def next(self, batch_size):
		"""retBatchSize <= batch_size
		Args:
			batch_size (int)
		Returns:
			np.ndarray: inputs, shape=[retBatchSize], dtype=numpy.int32
			np.ndarray: labels, shape=[retBatchSize], dtype=numpy.int32
			np.ndarray: weights, shape=[retBatchSize], dtype=numpy.float32
		"""
		if self.current_rank == self.DATA_SIZE:
			random.shuffle(self.rank_list)
			self.current_rank = 0

		batch = []  # [(etRank, ec_rank, w), ...]
		while len(batch) < batch_size and self.current_rank < self.DATA_SIZE:
			batch.extend(self.data[self.rank_list[self.current_rank]])
			self.current_rank += 1

		inputs, labels, weights = list(zip(*batch))
		return len(batch), np.array(inputs, dtype=np.int32), np.array(labels, dtype=np.int32), np.array(weights, dtype=np.float32)


def get_embed(encoder_name):
	"""
	Returns:
		np.ndarray: shape=[hpo_num, vec_size]
	"""
	encoder = HCEEncoder(encoder_name=encoder_name)
	return encoder.get_embed()


if __name__ == '__main__':

	encoder = HCEEncoder()
	c = HCEConfig(); c.vocab_size = 40
	encoder.build(c)









