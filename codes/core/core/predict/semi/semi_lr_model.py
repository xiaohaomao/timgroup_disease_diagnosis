

import tensorflow as tf
import os
import numpy as np
from scipy.sparse import vstack

from core.utils.constant import PHELIST_ANCESTOR, VEC_COMBINE_MEAN, MODEL_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING
from core.utils.constant import PREDICT_MODE
from core.utils.utils_tf import cross_entropy_with_probs
from core.utils.utils import get_logger, sparse_to_tuple
from core.reader.hpo_reader import HPOReader
from core.helper.data import BatchControllerMat, BatchControllerMixupMat, RandomGenBatchController, RGBCConfig, MultiBatchController
from core.helper.data.data_helper import DataHelper
from core.predict.ml_model.lr_neuron_model import LRNeuronModel, LRNeuronConfig


class SemiLRNeuronConfig(LRNeuronConfig):
	def __init__(self, d=None):
		super(SemiLRNeuronConfig, self).__init__()
		self.u_lambda = 0.5  # unlabel loss weight
		self.u_data_names = ('U_DEC',)
		self.min_hpo = 3
		self.del_dup = True
		if d is not None:
			self.assign(d)


class SemiLRModel(LRNeuronModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, embed_mat=None,
			combine_modes=(VEC_COMBINE_MEAN,), mode=PREDICT_MODE, model_name=None, init_para=True):
		super(SemiLRModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, model_name)
		self.name = 'SemiLRModel' if model_name is None else model_name
		self.SAVE_FOLDER = MODEL_PATH + os.sep + 'SemiLRModel' + os.sep + self.name
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)

		self.MODEL_PATH = self.SAVE_FOLDER + os.sep + 'model.ckpt'
		self.CONFIG_JSON = self.SAVE_FOLDER + os.sep + 'config'
		self.LOG_PATH = self.SAVE_FOLDER + os.sep + 'log'
		self.TEST_HISTORY_JSON = self.SAVE_FOLDER + os.sep + 'test_history.json'
		self.TRAIN_LOSS_FIG_PATH = self.SAVE_FOLDER + os.sep + 'train_loss.jpg'
		self.TEST_ACC_FIG_PATH = self.SAVE_FOLDER + os.sep + 'test_acc.jpg'

		self.SUMMARY_FOLDER = self.SAVE_FOLDER + os.sep + 'summary'
		os.makedirs(self.SUMMARY_FOLDER, exist_ok=True)

		if init_para and mode == PREDICT_MODE:
			self.restore()


	def get_loss(self, logits, c):
		y_ = tf.sparse.to_dense(self.placeholders['y_'])
		label_mask = self.placeholders['label_mask']
		label_logits = tf.boolean_mask(logits, label_mask)
		label_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=label_logits))

		unlabel_mask = tf.math.logical_not(label_mask)
		unlabel_logits = tf.boolean_mask(logits, unlabel_mask)
		p = tf.contrib.layers.softmax(unlabel_logits)
		unlabel_loss = cross_entropy_with_probs(p, p)

		loss = label_loss + c.u_lambda * unlabel_loss
		for var in self.vars.values():
			loss += c.w_decay * tf.nn.l2_loss(var)
		return loss


	def gen_placeholders(self, c):
		super(SemiLRModel, self).gen_placeholders(c)
		self.placeholders['label_mask'] = tf.placeholder(tf.bool, shape=(None,))


	def gen_feed_dict(self, c, bc):
		ldata, udata = bc.next_batch(c.batch_size)
		lX, ly_ = ldata   # sparseX and sparseY
		u_X = udata[0]  # sparseX
		X = vstack([lX, u_X])
		label_mask = np.zeros(shape=(X.shape[0],), dtype=np.bool); label_mask[:lX.shape[0]] = True
		X, x_val_shape = self.sparse_X_to_input(X)
		ret_dict = {
			self.placeholders['X']: X,
			self.placeholders['x_val_shape']: x_val_shape,  # int or None
			self.placeholders['y_']: sparse_to_tuple(ly_),
			self.placeholders['keep_prob']: c.keep_prob,
			self.placeholders['label_mask']: label_mask,
		}
		return ret_dict


	def get_batch_controller(self, c):
		label_bc = super(SemiLRModel, self).get_batch_controller(c)
		u_X, uy_ = DataHelper().getCombinedUTrainXY(c.u_data_names, self.phe_list_mode, self.vec_type, xdtype=np.float32,
													min_hpo=c.min_hpo, del_dup=c.del_dup, x_sparse=True)
		unlabel_bc = BatchControllerMat([u_X])
		bc = MultiBatchController([label_bc, unlabel_bc])
		return bc


def generate_model(vec_type=VEC_TYPE_0_1, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR,
				embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,), mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		SemiLRModel
	"""
	if mode == PREDICT_MODE:
		model = SemiLRModel(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, model_name)
		model.restore()
	else:
		model = SemiLRModel(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, model_name)
	return model


if __name__ == '__main__':
	model = generate_model(model_name='LRNeuTestModel')
	print(model.query(
		['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463']))  # OMIM:610253




