from core.utils.constant import OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_RMS, RELU, TANH, SIGMOID, ATT_MULTIPLY, ATT_ADD, ATT_DOT, SEED

import tensorflow as tf
import numpy as np

tf.set_random_seed(SEED)

def get_optimizer(type, lr):
	if type == OPTIMIZER_SGD:
		return tf.train.GradientDescentOptimizer(lr)
	if type == OPTIMIZER_ADAM:
		return tf.train.AdamOptimizer(lr)
	if type == OPTIMIZER_RMS:
		return tf.train.RMSPropOptimizer(lr)
	assert False


def get_active_func(func_name):
	if func_name == RELU:
		return tf.nn.relu
	elif func_name == TANH:
		return tf.tanh
	elif func_name == SIGMOID:
		return tf.sigmoid
	elif func_name is None:
		return lambda X: X
	else:
		assert False


def glorot(shape, name=None):
	"""Glorot & Bengio (AISTATS 2010) init.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/layers.py
	"""
	init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def zeros(shape, name=None):
	"""All zeros.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/layers.py
	"""
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
	"""Wrapper for tf.matmul (sparse vs dense).
	copy from https://github.com/tkipf/gcn/blob/master/gcn/layers.py
	"""
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res


def sparse_dropout(x, keep_prob, noise_shape):
	"""Dropout for sparse tensors.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/layers.py
	"""
	random_tensor = keep_prob
	random_tensor += tf.random_uniform(noise_shape)
	dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
	pre_out = tf.sparse_retain(x, dropout_mask)
	return pre_out * (1./keep_prob)


def dropout(X, keep_prob, x_val_shape=None):
	if type(X) == tf.SparseTensor:
		return sparse_dropout(X, keep_prob, x_val_shape)
	return tf.nn.dropout(X, keep_prob)


def cross_entropy_with_probs(labels, probs):
	return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(probs), axis=-1))


def euclid_dist2(h1, h2):
	return tf.reduce_sum(tf.square(h1 - h2), axis=-1)


def cosine_dist(h1, h2):
	h1_norm = tf.norm(h1, axis=-1)
	h2_norm = tf.norm(h2, axis=-1)
	return tf.reduce_sum(tf.multiply(h1, h2), axis=-1) / (h1_norm * h2_norm)


def dot_att_score(Q, K, scaled=True, *args, **kwargs):
	"""Note: d_q == d_k
	Args:
		Q (tf.Tensor): (N, T_q, d_q)
		K (tf.Tensor): (N, T_k, d_k)
	Returns:
		tf.Tensor: (N, T_q, T_k)
	"""
	A = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

	return A


def multiply_att_score(Q, K, Wa_name='Wa', *args, **kwargs):
	"""Note:
	Args:
		Q (tf.Tensor): (N, T_q, d_q)
		K (tf.Tensor): (N, T_k, d_k)
	Returns:
		tf.Tensor: (N, T_q, T_k)
	"""
	dq, dk = Q.shape[-1], K.shape[-1]
	Wa = tf.get_variable(
		Wa_name, shape=(dq, dk), dtype=tf.float32,
		initializer=tf.contrib.layers.xavier_initializer(uniform=True))
	return tf.matmul(tf.matmul(Q, Wa), tf.transpose(K, [0, 2, 1]))


def add_att_score(Q, K, dv, Wa_name='Wa', Ua_name='Ua', v_name='v', *args, **kwargs):
	"""
	Args:
		dv (int): dim of v
		Q (tf.Tensor): (N, T_q, d_q)
		K (tf.Tensor): (N, T_k, d_k)
	Returns:
		tf.Tensor: (N, T_q, T_k)
	"""
	dq, dk = Q.shape[-1], K.shape[-1]
	Wa = tf.get_variable(
		Wa_name, shape=(dq, dv), dtype=tf.float32,
		initializer=tf.contrib.layers.xavier_initializer(uniform=True)) # (d_q, d_v)
	Ua = tf.get_variable(
		Ua_name, shape=(dk, dv), dtype=tf.float32,
		initializer=tf.contrib.layers.xavier_initializer(uniform=True)) # (d_k, d_v)
	v = tf.get_variable(
		v_name, shape=(1, dv), dtype=np.float32,
		initializer=tf.contrib.layers.xavier_initializer(uniform=True)) # (1, dv)

	QW = tf.matmul(Q, Wa)   # (N, T_q, d_v)
	QW = tf.expand_dims(QW, axis=2) # (N, T_q, 1, d_v)
	QW = tf.tile(QW, [1, 1, K.shape[1], 1]) # (N, T_q, T_k, d_v)

	KU = tf.matmul(K, Ua)   # (N, T_k, d_v)
	KU = tf.expand_dims(KU, axis=1) # (N, 1, T_k, d_v)
	KU = tf.tile(KU, [1, Q.shape[1], 1, 1]) # (N, T_q, T_k, d_v)

	S = tf.matmul(v, tf.tanh(QW + KU))  # (N, T_q, T_k, 1)
	return tf.squeeze(S, axis=[-1]) # (N, T_q, T_k)


def mask_smat(S, seq_len, mask_value):
	"""mask Last Dim
	Args:
		S (tf.Tensor): (N, T_q, T_k)
		seq_len (tf.Tensor): (N,)
		mask_value (float): e.g. np.-inf for T_k; 0.0 for T_q
	Returns:
		tf.Tensor: masked S
	"""
	if mask_value != 0:
		mask = tf.sequence_mask(seq_len, tf.shape(S)[-1])    # (N, T_k)
		mask = tf.expand_dims(mask, 1)  # (N, 1, T_k)
		mask = tf.tile(mask, [1, tf.shape(S)[1], 1])  # (N, T_q, T_k)
		paddings = tf.ones_like(S) * mask_value  # (N, T_q, T_k)
		return tf.where(mask, S, paddings)

	mask = tf.sequence_mask(seq_len, tf.shape(S)[-1], dtype=tf.float32)  # (N, T_k)
	mask = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(S)[1], 1]) # (N, T_q, T_k)
	return S * mask


def attention_base(Q, K, V, q_seq_len, k_seq_len, att_score_func, prob_fn=tf.nn.softmax, score_mask=-2**32+1, scope='self_attention', *args, **kwargs):
	"""
	References:
		https://github.com/Kyubyong/transformer/blob/master/modules.py
	Args:
		Q (tf.Tensor): (N, T_q, d_q)
		K (tf.Tensor): (N, T_k, d_k)
		V (tf.Tensor): (N, T_k, d_v)
		q_seq_len (tf.Tensor): (N,)
		k_seq_len (tf.Tensor): (N,)
	Returns:
		tf.Tensor: (N, T_q, d_v)
	"""
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		S = att_score_func(Q, K, *args, **kwargs) # (N, T_q, T_k)
		S = mask_smat(S, k_seq_len, score_mask) # mask key
		P = prob_fn(S)   # (N, T_q, T_k)
		P = tf.transpose(mask_smat(tf.transpose(P, [0, 2, 1]), q_seq_len, 0), [0, 2, 1])   # (N, T_q, T_k)
		return tf.matmul(P, V)  # (N, T_q, d_v)


def dot_att(Q, K, V, q_seq_len, k_seq_len, scaled=True, prob_fn=tf.nn.softmax, score_mask=-2**32+1):
	return attention_base(Q, K, V, q_seq_len, k_seq_len, dot_att_score, prob_fn, score_mask, scaled=scaled)


def multiply_att(Q, K, V, q_seq_len, k_seq_len, Wa_name='Wa', prob_fn=tf.nn.softmax, score_mask=-2**32+1):
	return attention_base(Q, K, V, q_seq_len, k_seq_len, multiply_att_score, prob_fn, score_mask, Wa_name=Wa_name)


def add_att(Q, K, V, q_seq_len, k_seq_len, Wa_name, Ua_name, v_name, prob_fn=tf.nn.softmax, score_mask=-2**32+1):
	return attention_base(Q, K, V, q_seq_len, k_seq_len, add_att_score, prob_fn, score_mask, Wa_name=Wa_name, Ua_name=Ua_name, v_name=v_name)


def get_att_score_func(att_type):
	if att_type == ATT_DOT:
		return dot_att_score
	elif att_type == ATT_MULTIPLY:
		return multiply_att_score
	elif att_type == ATT_ADD:
		return add_att_score
	assert False


def multi_head_att(h_num, Q, K, V, q_seq_len, k_seq_len, att_type, prob_fn=tf.nn.softmax, score_mask=-2**32+1,
		scope='MultiHeadAttention', *args, **kwargs):
	"""
	References:
		https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py
		https://github.com/Kyubyong/transformer/blob/master/modules.py
	Args:
		Q (tf.Tensor): (N, T_q, d_model)
		K (tf.Tensor): (N, T_k, d_model)
		V (tf.Tensor): (N, T_k, d_model)
		q_seq_len (tf.Tensor): (N,)
		k_seq_len (tf.Tensor): (N,)
	Returns:
		tf.Tensor: (N, T_q, d_model)
	"""
	d_model = Q.get_shape().as_list()[-1]
	att_score_func = get_att_score_func(att_type)
	initializer = tf.contrib.layers.xavier_initializer()

	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		# Linear projections
		Q = tf.layers.dense(Q, d_model, use_bias=False, kernel_initializer=initializer)  # (N, T_q, d_model)
		K = tf.layers.dense(K, d_model, use_bias=False, kernel_initializer=initializer)  # (N, T_k, d_model)
		V = tf.layers.dense(V, d_model, use_bias=False, kernel_initializer=initializer)  # (N, T_k, d_model)

		# Split and concat
		Q_ = tf.concat(tf.split(Q, h_num, axis=2), axis=0)  # (h*N, T_q, d_model/h)
		K_ = tf.concat(tf.split(K, h_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)
		V_ = tf.concat(tf.split(V, h_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)

		# Scale q to prevent the dot product between q and k from growing too large.
		Q_ *= (d_model / h_num) ** -0.5

		q_seq_len = tf.expand_dims(q_seq_len, 1)    # (N, 1)
		q_seq_len = tf.tile(q_seq_len, [1, h_num])   # (N, h)
		q_seq_len = tf.reshape(q_seq_len, shape=(-1,))  # (N*h,)

		k_seq_len = tf.expand_dims(k_seq_len, 1)  # (N, 1)
		k_seq_len = tf.tile(k_seq_len, [1, h_num])  # (N, h)
		k_seq_len = tf.reshape(k_seq_len, shape=(-1,))  # (N*h,)

		# Attention
		new_V = attention_base(Q_, K_, V_, q_seq_len, k_seq_len, att_score_func, prob_fn, score_mask, 'self_attention', *args, **kwargs)  # (h*N, T_q, d_model/h)

		# Restore shape
		new_V = tf.concat(tf.split(new_V, h_num, axis=0), axis=2) # (N, T_q, d_model)



	return new_V


def ln(inputs, epsilon=1e-8, scope="ln"):
	'''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
	inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
	epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	scope: Optional scope for `variable_scope`.

	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]

		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
		gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
		normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
		outputs = gamma * normalized + beta

	return outputs


def ff(inputs, num_units, scope="ff"):
	'''position-wise feed forward net. See 3.3

	inputs: A 3d tensor with shape of [N, T, C].
	num_units: A list of two integers.
	scope: Optional scope for `variable_scope`.
	Returns:
	  A 3d tensor with the same shape and dtype as inputs
	'''
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		# Inner layer
		outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

		# Outer layer
		outputs = tf.layers.dense(outputs, num_units[1])

		# Residual connection
		outputs += inputs

		# Normalize
		outputs = ln(outputs)

	return outputs


def get_token_embedding(vocab_size, embed_size, name, zero_pad=True, pad_row=0, scope='embedding', pretrain_mat=None, trainable=True):
	"""
	Args:
		vocab_size (int): V
		embed_size (int): E
		zero_pad (bool)
		pad_row (int): which row to pad
		pretrain_mat (np.ndarray): (V, E)
	Returns:
		tf.Tensor: If zero_pad, then M.shape=(V+1, E) and M[pad_row]=[0, ..., 0]; or M.shape=(V, E)
	"""
	with tf.variable_scope(scope):
		embeddings = tf.get_variable(name, dtype=tf.float32, shape=(vocab_size, embed_size), trainable=trainable,
			initializer=tf.contrib.layers.xavier_initializer() if pretrain_mat is None else pretrain_mat)
		if zero_pad:
			embeddings = tf.concat( ( embeddings[:pad_row, :], tf.zeros(shape=[1, embed_size]), embeddings[pad_row:, :]), 0)
		return embeddings


if __name__ == '__main__':
	pass