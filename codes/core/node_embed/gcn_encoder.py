

from core.predict.config import Config
from core.node_embed.encoder import Encoder
from core.utils.constant import EMBEDDING_PATH, PHELIST_ANCESTOR, OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_RMS
from core.reader.hpo_reader import HPOReader
from core.draw.simpledraw import simple_line_plot
from core.utils.utils import get_logger, sparse_row_normalize, dense_row_normalize
from core.utils.utils import sparse_to_tuple, normalize_adj, preprocess_features, preprocess_adj
from core.utils.utils_tf import get_optimizer, dot, glorot, zeros, sparse_dropout
from core.helper.data.data_helper import DataHelper

import os
import shutil
import tensorflow as tf
from scipy.sparse import csr_matrix
import random
import numpy as np


class GCNConfig(Config):
	def __init__(self):
		super(GCNConfig, self).__init__()
		self.units = [16, 16]
		self.layer_norm = [None, None]
		self.acts = ['relu', None]
		self.keep_probs = [0.5, 0.5]
		self.bias = [False, False]
		self.weight_decays = [5e-6, 0.0]
		self.lr = 0.01
		self.epoch_num = 100
		self.embed_idx = 0   # -1: X | >=0: units
		self.summary_freq = 10

	def get_func(self, type):
		if type == 'l2LayerNorm':
			return lambda X: tf.nn.l2_normalize(X, 1)
		elif type == 'relu':
			return tf.nn.relu
		elif type == 'sigmoid':
			return tf.nn.sigmoid
		elif type is None:
			return lambda x: x
		else:
			return None


class GCNEncoder(Encoder):
	def __init__(self, hpo_reader=HPOReader(), encoder_name=None):
		super(GCNEncoder, self).__init__()
		self.hpo_reader = hpo_reader
		self.name = 'GCNEncoder' if encoder_name is None else encoder_name
		folder = EMBEDDING_PATH + os.sep + 'GCNEncoder'; os.makedirs(folder, exist_ok=True)
		folder = folder + os.sep + self.name; os.makedirs(folder, exist_ok=True)
		self.FOLDER = folder
		self.EMBEDDING_FOLDER = folder + os.sep + 'embedding'; os.makedirs(self.EMBEDDING_FOLDER, exist_ok=True)
		self.MODEL_PATH = folder + os.sep + 'model.ckpt'
		self.CONFIG_JSON = folder + os.sep + 'config'
		self.LOG_PATH = folder + os.sep + 'log'
		self.LOSS_FIG_PATH = folder + os.sep + 'loss.jpg'
		self.SUMMARY_FOLDER = folder + os.sep + 'summary'; os.makedirs(self.SUMMARY_FOLDER, exist_ok=True)
		self.hpo_embed = None    # np.ndarray; shape=(HPO_CODE_NUM, embed_size)
		self.vars = {} # {weight_i: tensor, 'bias_i': tensor, ...}
		self.placeholders = {}
		self.feed_dict = None
		self.embed_dict = {}

		self.hpo_num, self.dis_num = hpo_reader.get_hpo_num(), hpo_reader.get_dis_num()


	def get_X(self, c):
		raise NotImplementedError


	def get_loss(self, output, c):
		raise NotImplementedError


	def _get_embed(self, embed_idx=None, l2_norm=False):
		c = GCNConfig(); c.load(self.CONFIG_JSON)
		c.keep_probs = [1.0 for _ in c.keep_probs]
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		with tf.Session(config=sess_config) as sess:
			if not self.vars:
				self.build(c)
				self.load_vars(sess)
			self.feed_dict = None
			feed_dict = self.gen_feed_dict(c)
			embed_mat = self.get_embed_tensor(embed_idx if embed_idx is not None else c.embed_idx)
			if l2_norm:
				embed_mat = c.get_func('l2LayerNorm')(embed_mat)
			return sess.run(embed_mat, feed_dict=feed_dict)


	def get_embed(self, embed_idx=None, l2_norm=False):
		embedpath = self.EMBEDDING_FOLDER + '/embed_Idx{}_l2{}.npy'.format(embed_idx, l2_norm)
		if os.path.exists(embedpath):
			return np.load(embedpath)
		with tf.Graph().as_default():
			m = self._get_embed(embed_idx, l2_norm)
			np.save(embedpath, m)
			return m


	def get_embed_tensor(self, embed_idx):
		return self.embed_dict[self.get_var_name('embed', embed_idx)]


	def dropout_layer(self, X, keep_prob):
		if X is None:
			return X
		if type(X) == tf.SparseTensor:
			return sparse_dropout(X, keep_prob, self.placeholders['x_val_shape'])
		return tf.nn.dropout(X, keep_prob)


	def GCNLayer(self, X, norm_adj, W):
		"""
		Args:
			X (tf.Tensor or tf.SparseTensor or None): shape=(point_num, featureSize)
			norm_adj (tf.SparseTensor): shape=(point_num, point_num)
			W (tf.Tensor): shape=(input_dim, outputDim)
			sparseX (bool)
		"""
		if X is None:
			return dot(norm_adj, W, sparse=True)
		return dot(norm_adj, dot(X, W, sparse=(type(X)==tf.SparseTensor)), sparse=True)


	def get_var_name(self, name, layerNum):
		return '{}_{}'.format(name, layerNum)


	def get_norm_adj(self):
		"""
		Returns:
			tuple: tuple form of norm_adj
		"""
		return preprocess_adj(self.hpo_reader.get_hpo_adj_mat().astype(np.float32))


	def save_vars(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		saver.save(sess,  self.MODEL_PATH)


	def load_vars(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		saver.restore(sess, self.MODEL_PATH)


	def gen_placeholders(self, c):
		self.placeholders.update({
			'norm_adj': tf.sparse_placeholder(tf.float32, shape=(self.hpo_num, self.hpo_num)),
			'keep_probs': tf.placeholder(tf.float32, shape=(None,))
		})


	def gen_feed_dict(self, c):
		if self.feed_dict is not None:
			return self.feed_dict
		self.feed_dict = {
			self.placeholders['norm_adj']: self.get_norm_adj(),
			self.placeholders['keep_probs']: np.array(c.keep_probs, np.float32)
		}
		return self.feed_dict


	def build(self, c):
		self.gen_placeholders(c)
		norm_adj = self.placeholders['norm_adj']
		keep_probs = self.placeholders['keep_probs']

		X, input_dim, save_X = self.get_X(c)
		if save_X:
			self.vars[self.get_var_name('weight', -1)] = X
			self.embed_dict[self.get_var_name('embed', -1)] = X
		for i, outputDim in enumerate(c.units):
			w_name = self.get_var_name('weight', i)
			W = glorot((input_dim, outputDim), w_name)
			self.vars[w_name] = W
			X = self.dropout_layer(X, keep_probs[i])
			X = self.GCNLayer(X, norm_adj, W)    # GCN
			if c.bias[i]:
				b_name = self.get_var_name('bias', i)
				b = zeros((outputDim,), b_name)
				self.vars[b_name] = b
				X += b      # bias
			X = c.get_func(c.acts[i])(X)   # activate
			X = c.get_func(c.layer_norm[i])(X)    # Note: test
			self.embed_dict[self.get_var_name('embed', i)] = X
			input_dim = outputDim
		self.output = X

		self.loss = self.get_loss(X, c)
		optimizer = get_optimizer(OPTIMIZER_ADAM, c.lr)    #
		self.train_op = optimizer.minimize(self.loss)
		self.summary_op = self.summary(c)


	def get_parml2_norm(self, c):
		loss = 0
		for i in range(len(c.weight_decays)):
			loss += c.weight_decays[i] * tf.nn.l2_loss(self.vars[self.get_var_name('weight', i)])
			if c.bias[i]:
				loss += c.weight_decays[i] * tf.nn.l2_loss(self.vars[self.get_var_name('bias', i)])
		return loss


	def summary(self, c):
		tf.summary.scalar('loss', self.loss)
		for i in range(-1, len(c.units)):
			w_name = self.get_var_name('weight', i)
			if w_name not in self.vars: continue;
			tf.summary.histogram(w_name, self.vars[w_name])
		for i in range(-1, len(c.units)):
			embed_name = self.get_var_name('embed', i)
			if embed_name not in self.embed_dict: continue;
			tf.summary.histogram(embed_name, self.embed_dict[embed_name])
		return tf.summary.merge_all()


	def _train(self, c):
		logger = get_logger(self.name, log_path=self.LOG_PATH, mode='w')
		logger.info(self.name)
		logger.info(c)

		with tf.variable_scope(self.name):
			self.build(c)
			init_op = tf.global_variables_initializer()

		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		with tf.Session(config=sess_config, graph=tf.get_default_graph()) as sess:
			# print(tf.get_default_graph().get_operations())    # debug
			epoch_list, loss_list = [], []
			summary_writer = tf.summary.FileWriter(self.SUMMARY_FOLDER, sess.graph)
			sess.run(init_op)
			for i in range(c.epoch_num):
				feed_dict = self.gen_feed_dict(c)
				_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
				logger.info('Epoch {}({:.4}%): Batch Loss={}'.format(i, 100*i/c.epoch_num, loss))
				epoch_list.append(i); loss_list.append(loss)
				if i % c.summary_freq == 0:
					summ = sess.run(self.summary_op, feed_dict=feed_dict)
					summary_writer.add_summary(summ, i)
			self.save_vars(sess)
			c.save(self.CONFIG_JSON)
			self.draw_train_loss(self.LOSS_FIG_PATH, epoch_list, loss_list)
			self.draw_all_embed_norm(c, sess)


	def train(self, c):
		with tf.Graph().as_default():
			self._train(c)


	def draw_all_embed_norm(self, c, sess):
		for i in range(-1, len(c.units)):
			embed_name = self.get_var_name('embed', i)
			if embed_name not in self.embed_dict: continue;
			feed_dict = self.gen_feed_dict(c)
			feed_dict[self.placeholders['keep_probs']] = np.array([1.0 for _ in c.keep_probs], np.float32)
			self.draw_embed_hist(
				sess.run(self.embed_dict[embed_name], feed_dict=feed_dict),
				self.FOLDER+os.sep+embed_name+'.jpg'
			)


# =============================================================================
class GCNDisAsLabelConfig(GCNConfig):
	def __init__(self):
		super(GCNDisAsLabelConfig, self).__init__()
		self.xtype = 'I'    # 'I' | 'W'
		self.xsize = 64


class GCNDisAsLabelEncoder(GCNEncoder):
	def __init__(self, hpo_reader=HPOReader(), encoder_name=None):
		super(GCNDisAsLabelEncoder, self).__init__(hpo_reader, encoder_name)
		self.name = 'GCNDisAsLabelEncoder' if encoder_name is None else encoder_name


	def gen_placeholders(self, c):
		super(GCNDisAsLabelEncoder, self).gen_placeholders(c)
		self.placeholders.update({'hpo_dis_mat': tf.placeholder(tf.float32, shape=(self.hpo_num, self.dis_num))})


	def gen_feed_dict(self, c):
		if self.feed_dict is not None:
			return self.feed_dict
		self.feed_dict = super(GCNDisAsLabelEncoder, self).gen_feed_dict(c)
		hpo_dis_mat = DataHelper().get_train_X(sparse=False, dtype=np.float32).T
		self.feed_dict.update({self.placeholders['hpo_dis_mat']: hpo_dis_mat})
		return self.feed_dict


	def get_X(self, c):
		if c.xtype == 'I':
			return None, self.hpo_num, False
		if c.xtype == 'W':
			return glorot((self.hpo_num, c.xsize), 'X'), c.xsize, True
		assert False


	def get_loss(self, output, c):
		"""
		Args:
			output (tf.Tensor): shape=(hpo_num, hidden)
		"""
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.placeholders['hpo_dis_mat'], logits=output))
		loss += self.get_parml2_norm(c)
		if c.xtype == 'W':
			loss += c.weight_decays[0] * tf.nn.l2_loss(self.vars[self.get_var_name('weight', -1)])
		return loss


# =============================================================================
class GCNDisAsLabelFeatureConfig(GCNConfig):
	def __init__(self):
		super(GCNDisAsLabelFeatureConfig, self).__init__()


class GCNDisAsLabelFeatureEncoder(GCNEncoder):
	def __init__(self, hpo_reader=HPOReader(), encoder_name=None):
		super(GCNDisAsLabelFeatureEncoder, self).__init__(hpo_reader, encoder_name)
		self.name = 'GCNDisAsLabelFeatureEncoder' if encoder_name is None else encoder_name


	def gen_placeholders(self, c):
		super(GCNDisAsLabelFeatureEncoder, self).gen_placeholders(c)
		self.placeholders.update({
			'hpoDisMatLabel': tf.placeholder(tf.float32, shape=(self.hpo_num, self.dis_num)),
			'hpo_dis_mat_feature': tf.sparse_placeholder(tf.float32, shape=(self.hpo_num, self.dis_num)),
			'x_val_shape': tf.placeholder(tf.int32)
		})


	def gen_feed_dict(self, c):
		if self.feed_dict is not None:
			return self.feed_dict
		self.feed_dict = super(GCNDisAsLabelFeatureEncoder, self).gen_feed_dict(c)
		hpo_dis_mat = DataHelper().get_train_X(sparse=True, dtype=np.float32).T
		hpo_dis_mat_feature = sparse_to_tuple(sparse_row_normalize(hpo_dis_mat))
		self.feed_dict.update({
			self.placeholders['hpoDisMatLabel']: hpo_dis_mat.A,
			self.placeholders['hpo_dis_mat_feature']: hpo_dis_mat_feature,
			self.placeholders['x_val_shape']: hpo_dis_mat_feature[1].shape
		})
		return self.feed_dict


	def get_X(self, c):
		return self.placeholders['hpo_dis_mat_feature'], self.dis_num, False


	def get_loss(self, output, c):
		"""
		Args:
			output (tf.Tensor): shape=(hpo_num, hidden)
		"""
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.placeholders['hpoDisMatLabel'], logits=output))
		loss += self.get_parml2_norm(c)
		return loss


# =============================================================================
class GCNDisAsFeatureConfig(GCNConfig):
	def __init__(self):
		super(GCNDisAsFeatureConfig, self).__init__()
		self.batch_size = 32
		self.neg_sample_size = 32
		self.neg_weight = 10.0


class GCNDisAsFeatureEncoder(GCNEncoder):
	def __init__(self, hpo_reader=HPOReader(), encoder_name=None):
		super(GCNDisAsFeatureEncoder, self).__init__(hpo_reader, encoder_name)
		self.name = 'GCNDisAsFeatureEncoder' if encoder_name is None else encoder_name
		self.pos_sample_list = self.get_pos_sample_list()


	def get_pos_sample_list(self):
		"""
		Returns:
			list: [neighbors0, neighbors1, ...]; neighborsi=[hpo, hpo, ...]
		"""
		adj_mat = self.hpo_reader.get_hpo_adj_mat()
		hpo_num = self.hpo_reader.get_hpo_num()
		ret = [list(adj_mat[i].nonzero()[1]) for i in range(hpo_num)]
		return ret


	def gen_placeholders(self, c):
		super(GCNDisAsFeatureEncoder, self).gen_placeholders(c)
		self.placeholders.update({
			'hpo_dis_mat': tf.sparse_placeholder(tf.float32, shape=(self.hpo_num, self.dis_num)),
			# 'hpo_dis_mat': tf.placeholder(tf.float32, shape=(self.hpo_num, self.dis_num)),
			'outputs1_ids': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'outputs2_ids': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'neg_samples_ids': tf.placeholder(tf.int32, shape=(c.neg_sample_size,)),
			'x_val_shape': tf.placeholder(tf.int32)
		})


	def gen_feed_dict(self, c):
		if self.feed_dict is None:
			self.feed_dict = super(GCNDisAsFeatureEncoder, self).gen_feed_dict(c)
			hpo_dis_mat = DataHelper().get_train_X(sparse=True, dtype=np.float32).T
			hpo_dis_mat = sparse_to_tuple(sparse_row_normalize(hpo_dis_mat))
			# hpo_dis_mat = sparse_row_normalize(hpo_dis_mat)
			self.feed_dict.update({
				self.placeholders['hpo_dis_mat']: hpo_dis_mat,
				self.placeholders['x_val_shape']: hpo_dis_mat[1].shape
			})
		outputs1_ids, outputs2_ids, neg_samples_ids = self.get_samples(c)
		self.feed_dict.update({
			self.placeholders['outputs1_ids']: outputs1_ids,
			self.placeholders['outputs2_ids']: outputs2_ids,
			self.placeholders['neg_samples_ids']: neg_samples_ids,
		})
		return self.feed_dict


	def get_X(self, c):
		return self.placeholders['hpo_dis_mat'], self.dis_num, False


	def get_loss(self, output, c):
		outputs1 = tf.nn.embedding_lookup(output, self.placeholders['outputs1_ids'])
		outputs2 = tf.nn.embedding_lookup(output, self.placeholders['outputs2_ids'])
		neg_samples = tf.nn.embedding_lookup(output, self.placeholders['neg_samples_ids'])
		loss = self.xent_loss(outputs1, outputs2, neg_samples, c.neg_weight) / c.batch_size
		loss += self.get_parml2_norm(c)
		return loss


	def xent_loss(self, outputs1, outputs2, neg_samples, neg_weight):
		"""
		Args:
			outputs1 (tf.Tensor): shape=(batch_size, vec_size)
			outputs2 (tf.Tensor): shape=(batch_size, vec_size)
			neg_samples (tf.Tensor): shape=(negNum, vec_size)
			neg_weight (float)
		Returns:
			tf.Tensor: scalar
		"""
		outputs1 = tf.nn.l2_normalize(outputs1, 1)
		outputs2 = tf.nn.l2_normalize(outputs2, 1)
		neg_samples = tf.nn.l2_normalize(neg_samples, 1)
		aff = tf.reduce_sum(outputs1 * outputs2, axis=1)    # shape=(batch_size, 1)
		neg_aff = tf.matmul(outputs1, tf.transpose(neg_samples))  # shape=(batch_size, negNum)
		true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
		neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
		# loss = tf.reduce_sum(true_xent) + neg_weight * tf.reduce_sum(neg_xent)
		loss = tf.reduce_mean(true_xent) + neg_weight * tf.reduce_mean(neg_xent)
		return loss


	def get_samples(self, c):
		"""
		Returns:
			np.ndarray: outputs1_ids; np.array([hpo_int1, ...]), shape=(c.batch_size,)
			np.ndarray: outputs2_ids; np.array([hpo_int1, ...]), shape=(c.batch_size,)
			np.ndarray: neg_samples_ids; np.array([hpo_int1, ...]), shape=(c.neg_sample_size,)
		"""
		outputs1_ids = random.sample(range(self.hpo_reader.get_hpo_num()), c.batch_size)
		outputs2_ids = [random.sample(self.pos_sample_list[hpo_int], 1)[0] for hpo_int in outputs1_ids]
		hpo_num = self.hpo_reader.get_hpo_num()
		adj_mat = self.hpo_reader.get_hpo_adj_mat()
		outputs1_vec = csr_matrix(([1]*c.batch_size, ([0]*c.batch_size, outputs1_ids)), shape=(1, hpo_num), dtype=np.int32)
		true_lable_set = set((outputs1_vec * adj_mat).nonzero()[1])
		neg_samples_ids = random.sample(set(range(hpo_num)) - true_lable_set, c.neg_sample_size)
		return np.array(outputs1_ids, np.int32), np.array(outputs2_ids, np.int32), np.array(neg_samples_ids, np.int32)



def get_embed(encoder_name, encoder_class, embed_idx=None, l2_norm=False):
	"""
	Args:
		encoder_class (str): 'GCNDisAsLabelEncoder' | 'GCNDisAsLabelFeatureEncoder' | 'GCNDisAsFeatureEncoder'
	"""
	d = {'GCNDisAsLabelEncoder': GCNDisAsLabelEncoder, 'GCNDisAsLabelFeatureEncoder': GCNDisAsLabelFeatureEncoder, 'GCNDisAsFeatureEncoder': GCNDisAsFeatureEncoder}
	encoder = d[encoder_class](encoder_name=encoder_name)
	hpo_embed = encoder.get_embed(embed_idx, l2_norm)
	return hpo_embed


if __name__ == '__main__':
	pass




