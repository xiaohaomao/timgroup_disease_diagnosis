

from bert_syn.bert_pkg.bert.modeling import gelu
import tensorflow as tf

from bert_syn.utils.constant import RELU, LEAKY_RELU, TANH, SIGMOID, GELU

def get_sess_config():
	sess_config = tf.ConfigProto()
	# sess_config.gpu_options.allow_growth = True
	return sess_config


def get_active_func(func_name):
	if func_name == RELU:
		return tf.nn.relu
	elif func_name == LEAKY_RELU:
		return tf.nn.leaky_relu
	elif func_name == TANH:
		return tf.tanh
	elif func_name == SIGMOID:
		return tf.sigmoid
	elif func_name == GELU:
		return gelu
	elif func_name is None:
		return tf.identity
	else:
		assert False


def euclidean_dist2(X, Y):
	return tf.reduce_sum(tf.square(X - Y), axis=-1)


def euclidean_dist(X, Y, eps=0.0):
	"""range: (0, inf)
	Args:
		X (tf.tensor): (sample_num, feature_num)
		Y (tf.tensor): (sample_num, feature_num)
	Returns:
		tf.tensor: (sample_num,)
	"""
	return tf.sqrt(tf.reduce_sum(tf.square(X - Y), axis=-1) + eps)


def cosine_dist(X, Y):
	"""range: (0, 2)
	"""
	X = tf.math.l2_normalize(X, axis=-1)
	Y = tf.math.l2_normalize(Y, axis=-1)
	return 1. - tf.reduce_sum(tf.multiply(X, Y), axis=-1)


if __name__ == '__main__':
	pass

