import json
from time import time
from copy import deepcopy
import numpy as np
import tensorflow as tf
import random
import os

from core.predict.config import Config
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.node_embed.encoder import Encoder
from core.utils.constant import DIS_SIM_JACCARD, DIS_SIM_MICA, OPTIMIZER_ADAM, OPTIMIZER_SGD, SIM_TO_DIST_E, EMBEDDING_PATH
from core.utils.utils_tf import get_active_func, euclid_dist2, cosine_dist, get_optimizer
from core.reader.hpo_reader import HPOReader
from core.helper.data.batch_controller import BatchController, BatchControllerObjList
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE
from core.utils.utils import flatten, get_all_descendents_with_dist, modify_ary_ele_with_dict, get_logger, check_return, padding, combine_embed
from core.helper.data.data_helper import DataHelper
from core.predict.calculator.ic_calculator import get_hpo_IC_vec

EUCLIDIAN_DIST2 = 'EUCLIDIAN_DIST2'
COSINE_DIST = 'COSINE_DIST'

W_SIGMOID = 'W_SIGMOID'
W_LINEAR = 'W_LINEAR'

COMBINE_HPO_AVG = 'COMBIND_HPO_AVG'
COMBINE_HPO_IC_WEIGHT = 'COMBIND_HPO_IC_WEIGHT'


class GDJointConfig(Config):
	def __init__(self, d=None):
		super(GDJointConfig, self).__init__()
		self.batch_size = 256
		self.optimizer = OPTIMIZER_ADAM
		self.lr = 0.001
		self.total_epoch = 10000
		self.w_decay = 0.0

		self.n_features = 256
		self.f_units = []
		self.f_actives = []

		self.use_W = True
		self.w_type = W_SIGMOID
		self.beta = 0.5
		self.gamma = 6

		self.alpha1 = 1.0
		self.use_dis_dist = True
		self.dis_sim_type = DIS_SIM_MICA
		self.sim_to_dist = SIM_TO_DIST_E
		self.dis_dist_min = 0.1
		self.embed_dist = EUCLIDIAN_DIST2

		self.d_combine_hpo = COMBINE_HPO_AVG
		self.phe_list_mode = PHELIST_REDUCE

		self.alpha2 = 1.0
		self.co_phe_list_mode = PHELIST_REDUCE
		self.co_W_type = W_SIGMOID
		self.co_W_beta = 1.0
		self.co_W_gamma = 2

		if d is not None:
			self.assign(d)


class GDJointEncoder(Encoder):
	def __init__(self, hpo_reader=HPOReader(), encoder_name=None):
		super(GDJointEncoder, self).__init__()
		self.name = 'GDJointEncoder' if encoder_name is None else encoder_name
		self.hpo_reader = hpo_reader
		self.HPO_NUM, self.DIS_NUM = hpo_reader.get_hpo_num(), hpo_reader.get_dis_num()
		self.x_id_array_to_hid_array = np.vectorize(lambda a, xIdMapHId: xIdMapHId[a])
		self.hpo_IC_vec = None
		self.hpo_embed = None
		self.dis_embed = None

		self.SAVE_FOLDER = EMBEDDING_PATH + os.sep + 'GDJointEncoder' + os.sep + self.name; os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.EMBED_NPY_PATH = self.SAVE_FOLDER + os.sep + 'embedding.npy'
		self.EMBED_TXT_PATH = self.SAVE_FOLDER + os.sep + 'embedding.txt'
		self.CONFIG_JSON = self.SAVE_FOLDER + os.sep + 'config'
		self.LOG_PATH = self.SAVE_FOLDER + os.sep + 'log'
		self.LOSS_FIG_PATH = self.SAVE_FOLDER + os.sep + 'loss.jpg'
		self.EMBED_ELE_FIG_PATH = self.SAVE_FOLDER + os.sep + 'ele_distribution.jpg'
		self.EMBED_L2_FIG_PATH = self.SAVE_FOLDER + os.sep + 'l2_distribution.jpg'


	@check_return('hpo_embed')
	def get_embed(self):
		return np.load(self.EMBED_NPY_PATH)


	def get_config(self):
		return GDJointConfig(json.load(open(self.CONFIG_JSON)))


	def build(self, c):
		self.gen_placeholders(c)
		x_mat = tf.get_variable('x_mat', shape=(self.HPO_NUM, c.n_features), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
		x_id_seq = self.placeholders['x_id_seq']
		X = tf.nn.embedding_lookup(x_mat, x_id_seq)
		H = self.f(X, c)
		self.embed_mat = H

		hIdSeq1, hIdSeq2, w = self.placeholders['hIdSeq1'], self.placeholders['hIdSeq2'], self.placeholders['w']
		h1, h2 = tf.nn.embedding_lookup(H, hIdSeq1), tf.nn.embedding_lookup(H, hIdSeq2)
		loss1 = self.get_loss1(h1, h2, w, c)

		dHIdSeqs1, d_hid_seqs2 = self.placeholders['dHIdSeqs1'], self.placeholders['d_hid_seqs2']
		dHIdLen1, d_hid_len2 = self.placeholders['dHIdLen1'], self.placeholders['d_hid_len2']
		d_dist = self.placeholders['d_dist']  # (batch_size,)
		loss2 = self.get_loss2(H, dHIdSeqs1, dHIdLen1, d_hid_seqs2, d_hid_len2, d_dist, c)

		coHIdSeq1, coHIdSeq2, co_W = self.placeholders['coHIdSeq1'], self.placeholders['coHIdSeq2'], self.placeholders['co_W']
		coH1, co_H2 = tf.nn.embedding_lookup(H, coHIdSeq1), tf.nn.embedding_lookup(H, coHIdSeq2)
		loss3 = self.get_loss3(coH1, co_H2, co_W, c)
		reg_loss = tf.nn.l2_loss(x_mat)
		self.loss = loss1 + c.alpha1 * loss2 + c.alpha2 * loss3 + c.w_decay * reg_loss
		self.global_step = tf.Variable(0, trainable=False)
		optimizer = get_optimizer(c.optimizer, c.lr)
		self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		self.init_op = tf.global_variables_initializer()

		self.loss1 = loss1
		self.loss2 = loss2
		self.loss3 = loss3
		self.reg_loss = reg_loss


	def clip_grad(self, optimizer, loss, max_norm, gs):
		grads_and_vars = optimizer.compute_gradients(loss)
		gradients, variables = zip(*grads_and_vars)  # unzip
		gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
		grads_and_vars = zip(gradients, variables)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=gs)
		return train_op


	def f(self, X, c):
		"""output embedding
		Args:
			X (tf.Tensor): shape = (None, n_features)
		Returns:
			tf.Tensor: shape = (None, emebedSize)
		"""
		for i in range(len(c.f_units)):
			W = tf.get_variable(
				'W_{}'.format(i+1), shape=(X.shape[1], c.f_units[i]), dtype=tf.float32,
				initializer=tf.contrib.layers.xavier_initializer(uniform=True))
			b = tf.get_variable(
				'b_{}'.format(i+1), shape=(c.f_units[i],), dtype=tf.float32,
				initializer=tf.zeros_initializer()
			)
			X = tf.matmul(X, W) + b
			X = get_active_func(c.f_actives[i])(X)
		return X


	def vec_dist(self, h1, h2, c):
		if c.embed_dist == EUCLIDIAN_DIST2:
			return euclid_dist2(h1, h2)
		elif c.embed_dist == COSINE_DIST:
			return cosine_dist(h1, h2)
		assert False


	def get_loss1(self, h1, h2, w, c):
		"""
		Args:
			h1 (tf.Tensor): shape=(batch_size, embed_size)
			h2 (tf.Tensor): shape=(batch_size, embed_size)
			w (tf.Tensor): shape=(batch_size)
		"""
		return tf.reduce_mean(self.vec_dist(h1, h2, c) * w)


	def get_loss2(self, embed_mat, dHPOIdSeqs1, d_X_id_len1, dHPOIdSeqs2, d_X_id_len2, d_dist, c):
		"""
		Args:
			distMat (tf.Tensor,): (batch_size)
		"""
		d1 = self.get_dis_embed(embed_mat, dHPOIdSeqs1, d_X_id_len1, c.d_combine_hpo)
		d2 = self.get_dis_embed(embed_mat, dHPOIdSeqs2, d_X_id_len2, c.d_combine_hpo)

		return tf.reduce_mean( tf.abs(self.vec_dist(d1, d2, c) - tf.square(d_dist)) )


	def get_loss3(self, h1, h2, co_W, c):
		return tf.reduce_mean(self.vec_dist(h1, h2, c) * co_W)


	def get_dis_embed(self, embed_mat, dHPOIdSeqs, dHPONum, d_combine_hpo):
		"""
		Args:
			embed_mat (tf.Tensor or np.ndarray): shape=(None, embed_size)
			dHPOIdSeqs (tf.Tensor or np.ndarray): shape=(batch_size, max_len)
			dHPONum (tf.Tensor or np.ndarray): shape=(batch_size,)
		Returns:
			tf.Tensor: shape=(batch_size, embed_size)
		"""
		mask = tf.sequence_mask(dHPONum, tf.shape(dHPOIdSeqs)[1], dtype=tf.float32)
		m = tf.nn.embedding_lookup(embed_mat, dHPOIdSeqs)
		if d_combine_hpo == COMBINE_HPO_AVG:
			wgt = tf.cast(1 / tf.reshape(dHPONum, (-1, 1)), tf.float32)
			return tf.reduce_sum(m * tf.expand_dims(mask, -1) * tf.expand_dims(wgt, -1), axis=1)
		elif d_combine_hpo == COMBINE_HPO_IC_WEIGHT:
			wgt = tf.nn.embedding_lookup(self.placeholders['IC_vec'], dHPOIdSeqs) * mask
			wgt = wgt / tf.reshape(tf.reduce_sum(wgt, axis=-1), (-1, 1))
			return tf.reduce_sum(m * tf.expand_dims(wgt, -1), axis=1)
		assert False


	def phe_lists_to_embed_mat(self, embed_mat, hpo_int_lists, d_combine_hpo):
		"""
		Args:
			embed_mat (np.ndarray): (hpo_num, embed_size)
			hpo_id_seqs (np.ndarray or list): [[hpo_int1, hpo_int2, ...], ...]
		Returns:
			np.ndarray: (sample_num, embed_size)
		"""
		feed_dict = {}
		if d_combine_hpo == COMBINE_HPO_IC_WEIGHT:
			feed_dict['IC_vec'] = self.get_hpo_IC_vec()
		hpo_id_seqs, seq_len = padding(hpo_int_lists, padwith=0)
		with tf.Session() as sess:
			return sess.run(self.get_dis_embed(embed_mat, hpo_id_seqs, seq_len, d_combine_hpo))


	def gen_placeholders(self, c):
		self.placeholders = {
			'x_id_seq': tf.placeholder(tf.int32, shape=(None,)),
			'hIdSeq1': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'hIdSeq2': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'w': tf.placeholder(tf.float32, shape=(c.batch_size,)),

			'dHIdSeqs1': tf.placeholder(tf.int32, shape=(c.batch_size, None)),
			'd_hid_seqs2': tf.placeholder(tf.int32, shape=(c.batch_size, None)),
			'dHIdLen1': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'd_hid_len2': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'd_dist': tf.placeholder(tf.float32, shape=(c.batch_size,)),
			'IC_vec': tf.placeholder(tf.float32, shape=(None,)),

			'coHIdSeq1': tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'coHIdSeq2':tf.placeholder(tf.int32, shape=(c.batch_size,)),
			'co_W':tf.placeholder(tf.float32, shape=(c.batch_size,)),
		}


	def get_id_seq(self, *x_id_seqList):
		return np.unique(np.hstack(x_id_seqList)).astype(np.int32)


	def gen_feed_dict(self, c, hpo_bc, dis_Bc, co_hpo_bc):
		x_id_seq1, x_id_seq2, depth = hpo_bc.next_batch(c.batch_size)
		w = self.depth_to_w(depth, c)

		dis_id_seq1, dX_id_seqs1, d_X_id_len1, dis_id_seq2, dX_id_seqs2, d_X_id_len2 = dis_Bc.next_batch(c.batch_size)
		d_dist = self.get_dis_dist(dis_id_seq1, dis_id_seq2, c) * 10.0

		co_X_id_seq1, co_X_id_seq1, co_ary = co_hpo_bc.next_batch(c.batch_size)
		co_W = self.co_occur_to_W(co_ary, c)

		all_X_id_seq = self.get_id_seq(x_id_seq1, x_id_seq2, dX_id_seqs1.flatten(), dX_id_seqs2.flatten(), co_X_id_seq1, co_X_id_seq1)
		x_id_to_hid_ary = np.zeros((all_X_id_seq.max() + 1), dtype=np.int32)
		x_id_to_hid_ary[all_X_id_seq] = np.arange(0, all_X_id_seq.shape[0])

		ret_dict = {
			self.placeholders['x_id_seq']: all_X_id_seq,
			self.placeholders['hIdSeq1']: x_id_to_hid_ary[x_id_seq1],
			self.placeholders['hIdSeq2']:x_id_to_hid_ary[x_id_seq2],
			self.placeholders['w']: w,

			self.placeholders['dHIdSeqs1']: x_id_to_hid_ary[dX_id_seqs1.flatten()].reshape(dX_id_seqs1.shape),
			self.placeholders['d_hid_seqs2']: x_id_to_hid_ary[dX_id_seqs2.flatten()].reshape(dX_id_seqs2.shape),
			self.placeholders['dHIdLen1']: d_X_id_len1,
			self.placeholders['d_hid_len2']: d_X_id_len2,
			self.placeholders['d_dist']: d_dist,

			self.placeholders['coHIdSeq1']: x_id_to_hid_ary[co_X_id_seq1],
			self.placeholders['coHIdSeq2']:x_id_to_hid_ary[co_X_id_seq1],
			self.placeholders['co_W']: co_W,

		}
		if c.d_combine_hpo == COMBINE_HPO_IC_WEIGHT:
			ret_dict[self.placeholders['IC_vec']] = self.hpo_IC_vec[all_X_id_seq]
		return ret_dict


	def get_dis_dist(self, dis_ints1, dis_ints2, c):
		"""
		Returns:
			np.ndarray: (batch_size,)
		"""
		assert len(dis_ints1) == len(dis_ints2)
		if not c.use_dis_dist:
			return np.ones(shape=(len(dis_ints1,)), dtype=np.float32)
		dis_dist_ary = self.dis_dist_mat[(dis_ints1, dis_ints2)]
		dis_dist_ary[dis_dist_ary < c.dis_dist_min] = c.dis_dist_min
		return dis_dist_ary


	def w_to_norm(self, w, w_type=None, beta=None, gamma=None):
		if w_type == W_SIGMOID:
			return 2 / (1 + np.exp(-beta * w)) - 1
		elif w_type == W_LINEAR:
			w = w / gamma
			w[w > 1] = 1.0
			return w
		assert False


	def depth_to_w(self, d, c):
		"""
		Args:
			d (np.ndarray): (batch_size,)
		"""
		if not c.use_W:
			return np.ones(d.shape[0], dtype=np.float32)
		return self.w_to_norm(d, c.w_type, c.beta, c.gamma)


	def co_occur_to_W(self, n, c):
		"""
		Args:
			n (np.ndarray): (batch_size,)
		"""
		return self.w_to_norm(n, c.co_W_type, c.co_W_beta, c.co_W_gamma)


	@check_return('hpo_IC_vec')
	def get_hpo_IC_vec(self):
		return get_hpo_IC_vec(self.hpo_reader, default_IC=0.0)


	def _train(self, c):
		if c.use_dis_dist:
			dsc = DisSimCalculator()
			self.dis_dist_mat = dsc.sim_mat_to_dist_mat(dsc.get_dis_sim_mat(c.dis_sim_type), c.sim_to_dist)
		if c.d_combine_hpo == COMBINE_HPO_IC_WEIGHT:
			self.get_hpo_IC_vec()
		dis_Bc = DisBatchController(self.hpo_reader, c.phe_list_mode, padding_id=0)
		hpo_bc = HPOBatchController(self.hpo_reader)
		co_hpo_bc = CoHPOBatchController(self.hpo_reader, c.co_phe_list_mode)
		self.build(c)

		logger = get_logger(self.name)
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		with tf.Session(config=sess_config) as sess:
			epoch_list, loss_list = [], []
			sess.run(self.init_op)
			for i in range(1, c.total_epoch+1):
				feed_dict = self.gen_feed_dict(c, hpo_bc, dis_Bc, co_hpo_bc)
				_, loss, loss1, loss2, loss3, gs = sess.run([
					self.train_op, self.loss, self.loss1, self.loss2, self.loss3, self.global_step], feed_dict=feed_dict)
				if i % 50 == 0:
					logger.info('Epoch {}({:.4}%): Batch Loss={}, loss1={}, loss2={}, loss3={}'.format(i, 100*i/c.total_epoch, loss, loss1, loss2, loss3))
					epoch_list.append(i); loss_list.append(loss)
					self.draw_train_loss(self.LOSS_FIG_PATH, epoch_list, loss_list)
			self.hpo_embed = sess.run(self.embed_mat, feed_dict={self.placeholders['x_id_seq']: np.arange(0, self.HPO_NUM, dtype=np.int32)})
			self.save(self.hpo_embed, c)
			self.draw_embed_hist(self.hpo_embed, self.EMBED_ELE_FIG_PATH, l2_norm=False)
			self.draw_embed_hist(self.hpo_embed, self.EMBED_L2_FIG_PATH, l2_norm=True)


	def train(self, c):
		self.g = tf.Graph()
		with self.g.as_default():
			self._train(c)


	def save(self, embed_mat, c):
		self.save_embed_txt(embed_mat, self.EMBED_TXT_PATH)
		np.save(self.EMBED_NPY_PATH, embed_mat)
		c.save(self.CONFIG_JSON)


class HPOBatchController(BatchControllerObjList):
	def __init__(self, hpo_reader=HPOReader()):
		self.hpo_reader = hpo_reader
		HPO_NUM = self.hpo_reader.get_hpo_num()

		hpo2depth = get_all_descendents_with_dist('HP:0000001', self.hpo_reader.get_slice_hpo_dict())
		hpo_list = self.hpo_reader.get_hpo_list()
		self.hpo_depth = np.array([hpo2depth[hpo] for hpo in hpo_list], dtype=np.int32)

		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		self.parent_num = np.array([len(hpo_int_dict[i].get('IS_A', [])) for i in range(HPO_NUM)], dtype=np.int32)
		self.parent_id_mat = np.zeros(shape=(HPO_NUM, self.parent_num.max()), dtype=np.int32)
		for i in range(1, HPO_NUM):
			self.parent_id_mat[i, :self.parent_num[i]] = hpo_int_dict[i]['IS_A']

		data = list(range(1, HPO_NUM))
		super(HPOBatchController, self).__init__(data)


	def next_batch(self, batch_size):
		"""
		Returns:
			np.ndarray: child HPO Ids; shape=(batch_size,)
			np.ndarray: parent HPO Ids; shape=(batch_size,)
			np.ndarray: depth of parent + 1; shape=(batch_size,)
		"""
		chd_id_seq = np.array(super(HPOBatchController, self).next_batch(batch_size), dtype=np.int32)
		pat_id_seq = self.parent_id_mat[chd_id_seq, (np.random.sample((batch_size,)) * self.parent_num[chd_id_seq]).astype(np.int32)]
		chd_depth = self.hpo_depth[pat_id_seq] + 1
		return chd_id_seq, pat_id_seq, chd_depth


class CoHPOBatchController(BatchController):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE):
		self.hpo_reader = hpo_reader
		self.co_mat = hpo_reader.get_hpo_co_mat(phe_list_mode, dtype=np.int32)
		self.data = np.array([i for i in range(self.co_mat.shape[0]) if self.co_mat[i].count_nonzero() > 0])

		super(CoHPOBatchController, self).__init__(len(self.data))


	def next_batch(self, batch_size):
		"""
		Returns:
			np.ndarray: colIdAry; (batch_size,)
			np.ndarray: colIdAry; (batch_size)
			np.ndarray: co-occur times; (batch_size,)
		"""
		hpo_id_seq1 = self.data[super(CoHPOBatchController, self).next_batch(batch_size)]
		hpo_id_seq2 = np.array([np.random.choice(self.co_mat[row].nonzero()[1], 1)[0] for row in hpo_id_seq1], np.int32)
		co_ary = self.co_mat[hpo_id_seq1, hpo_id_seq2].A1
		return hpo_id_seq1, hpo_id_seq2, co_ary


class DisBatchController(BatchController):
	def __init__(self, hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, padding_id=0):
		self.hpo_reader = hpo_reader
		self.padding_id = padding_id
		self.dh = DataHelper(self.hpo_reader)
		raw_X, _ = self.dh.get_train_raw_Xy(phe_list_mode)
		self.dX_id_seqs, self.d_X_id_len = padding(raw_X, padwith=self.padding_id)
		super(DisBatchController, self).__init__(self.dX_id_seqs.shape[0])


	def _next_batch(self, batch_size):
		"""
		Returns:
			np.ndarray: dis_id_seq, shape=(batch_size,)
			np.ndarray: dX_id_seqs, shape=(batch_size, max_len)
			np.ndarray: d_X_id_len, shape=(batch_size,)
		"""
		dis_id_seq = super(DisBatchController, self).next_batch(batch_size)
		dX_id_seqs, d_X_id_len = self.dX_id_seqs[dis_id_seq], self.d_X_id_len[dis_id_seq]
		return dis_id_seq, dX_id_seqs, d_X_id_len


	def next_batch(self, batch_size):
		"""
		Returns:
			np.ndarray: dis_id_seq1, shape=(batch_size,)
			np.ndarray: dX_id_seqs1, shape=(batch_size, maxLen1)
			np.ndarray: d_X_id_len1, shape=(batch_size,)

			np.ndarray: dis_id_seq2, shape=(batch_size,)
			np.ndarray: dX_id_seqs2, shape=(batch_size, maxLen2)
			np.ndarray: d_X_id_len2, shape=(batch_size,)
		"""
		dis_id_seq1, dX_id_seqs1, d_X_id_len1 = self._next_batch(batch_size)
		dis_id_seq2, dX_id_seqs2, d_X_id_len2 = self._next_batch(batch_size)
		return dis_id_seq1, dX_id_seqs1, d_X_id_len1, dis_id_seq2, dX_id_seqs2, d_X_id_len2


def get_embed(encoder_name):
	"""
	Returns:
		np.ndarray: shape=[hpo_num, vec_size]
	"""
	encoder = GDJointEncoder(encoder_name=encoder_name)
	return encoder.get_embed()


def phe_lists_to_embed_mat(embed_mat, hpo_int_lists, d_combine_hpo):
	if d_combine_hpo == COMBINE_HPO_AVG:
		return combine_embed(embed_mat, hpo_int_lists, combine_mode='avg')
	elif d_combine_hpo == COMBINE_HPO_IC_WEIGHT:
		# FIXME: get_hpo_IC_vec should use correct HPOReader; assert False here
		assert False
		return combine_embed(embed_mat, hpo_int_lists, combine_mode='weight', id_weights=get_hpo_IC_vec(default_IC=0.0))

if __name__ == '__main__':
	pass

