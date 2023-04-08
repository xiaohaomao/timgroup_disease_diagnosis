import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
import shutil
import json
from copy import deepcopy
import scipy, random

from core.predict.ensemble.ordered_multi_model import OrderedMultiModel
from core.predict.ensemble.random_model import RandomModel
from core.predict.model import TensorflowModel
from core.predict.config import Config
from core.utils.constant import PHELIST_ANCESTOR, VEC_COMBINE_MEAN, OPTIMIZER_ADAM, MODEL_PATH, VEC_TYPE_0_1, VEC_TYPE_EMBEDDING
from core.utils.constant import PREDICT_MODE, TRAIN_MODE, TEST_DATA, VALIDATION_DATA, VALIDATION_TEST_DATA, HYPER_TUNE_RANK_SCORE, HYPER_TUNE_Z_SCORE, HYPER_TUNE_AVE_SCORE, SEED
from core.utils.constant import get_tune_data_names, get_tune_metric_weights, get_tune_metric_names, get_tune_data_weights
from core.utils.utils_tf import glorot, dot, get_optimizer, zeros, dropout
from core.utils.utils import get_logger, sparse_to_tuple, dict_list_add, delete_logger, timer
from core.reader import HPOReader, HPOFilterDatasetReader
from core.helper.data import BatchControllerMat, BatchControllerMixupMat, RandomGenBatchController, RGBCConfig
from core.predict.model_testor import ModelTestor
from core.draw.simpledraw import simple_multi_line_plot
from core.helper.data.data_helper import DataHelper
from core.helper.hyper.hyper_tune_helper import HyperTuneHelper

class LRNeuronConfig(Config):
	def __init__(self, d=None):
		super(LRNeuronConfig, self).__init__()
		self.batch_size = 512
		self.keep_prob = 1.0
		self.w_decay = 0.0
		self.optimizer = OPTIMIZER_ADAM
		self.lr = 0.001
		self.max_epoch_num = 600
		self.min_epoch_num = 150

		self.early_stop_patience = 10
		self.mt_draw_save_freq = 500
		self.perturbation = False
		self.pertur_weight = [0.5, 0.2, 0.2, 0.05, 0.05]
		self.mixup = False
		self.mix_alpha = 1.0
		self.dis_expand_ances = False
		self.multi_label = False
		self.desc_dp = None

		self.eval_data = VALIDATION_DATA
		self.print_freq = 10
		self.summary = False
		self.summary_freq = 200
		self.mt_test_freq = 200
		self.val_freq = 200
		self.val_data_names = get_tune_data_names(self.eval_data)
		self.test_acc_ats = [1, 10]
		self.mt_val_data_names = get_tune_data_names(self.eval_data)
		self.mt_val_metrics = get_tune_metric_names()
		self.val_rank_weights = [get_tune_data_weights(self.eval_data), get_tune_metric_weights()]
		self.hyper_score_type = HYPER_TUNE_RANK_SCORE
		self.mt_draw_data_names = self.mt_val_data_names

		self.mt_test_data_names = get_tune_data_names(TEST_DATA)
		self.mt_test_metrics = get_tune_metric_names()
		self.test_rank_weights = [get_tune_data_weights(TEST_DATA), get_tune_metric_weights()]

		self.mt_draw_metrics = []
		self.seed = SEED

		self.n_init = 0

		self.n_features = None
		self.class_num = None

		if d is not None:
			self.assign(d)


class LRNeuronModel(TensorflowModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, embed_mat=None,
			combine_modes=(VEC_COMBINE_MEAN,), mode=PREDICT_MODE, model_name=None, save_folder=None, init_para=True,
			use_rd_mix_code=False):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(LRNeuronModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, use_rd_mix_code)
		self.name = 'LRNeuronModel' if model_name is None else model_name
		self.init_save_folder(save_folder)

		self.placeholders = {}
		self.vars = {}
		self.summary_dict = {}
		self.metric_dict = {}
		self.history_helper = None

		self.mode = mode
		if init_para and mode == PREDICT_MODE:
			self.restore()


	@timer
	def __del__(self):
		if hasattr(self, 'sess') and self.sess is not None:
			self.sess.close()


	def init_save_folder(self, save_folder=None):
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'LRNeuronModel', self.name) if save_folder is None else save_folder
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_PATH = os.path.join(self.SAVE_FOLDER, 'model.ckpt')
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, 'config.json')
		self.LOG_PATH = os.path.join(self.SAVE_FOLDER, 'log.txt')
		self.VAL_HISTORY_FOLDER = os.path.join(self.SAVE_FOLDER, 'val_history')
		self.HISTORY_LOSS_JSON = os.path.join(self.SAVE_FOLDER, 'loss.json')
		self.HISTORY_LOSS_FIG_PATH = os.path.join(self.SAVE_FOLDER, 'loss.jpeg')
		self.PERFORMANCE_FIG_PATH = os.path.join(self.SAVE_FOLDER, 'performance.jpeg')
		self.SUMMARY_FOLDER = os.path.join(self.SAVE_FOLDER, 'summary')
		os.makedirs(self.SUMMARY_FOLDER, exist_ok=True)


	def sparse_X_to_input(self, X):
		X = sparse_to_tuple(X)
		return X, X[1].shape


	def gen_placeholders(self, c):
		self.placeholders = {
			'y_': tf.sparse_placeholder(tf.float32, shape=(None, c.class_num)),
			'keep_prob': tf.placeholder(tf.float32),
			'seqy_': tf.placeholder(tf.int64, shape=(None, 1))
		}
		if self.vec_type == VEC_TYPE_EMBEDDING:
			self.placeholders['X'] = tf.placeholder(tf.float32, shape=(None, c.n_features))
		else:
			self.placeholders['X'] = tf.sparse_placeholder(tf.float32, shape=(None, c.n_features))
			self.placeholders['x_val_shape'] = tf.placeholder(tf.int32)


	def gen_feed_dict(self, c, bc):
		X, y_ = bc.next_batch(c.batch_size)
		X, x_val_shape = self.sparse_X_to_input(X)
		ret_dict = {
			self.placeholders['X']: X,
			self.placeholders['x_val_shape']: x_val_shape,
			self.placeholders['y_']: sparse_to_tuple(y_),
			self.placeholders['keep_prob']: c.keep_prob,
		}
		return ret_dict


	def get_single_batch_controller(self, c):
		x_sparse = (self.vec_type != VEC_TYPE_EMBEDDING)
		if c.perturbation:
			bc_config = RGBCConfig({
				'raw_xy':False, 'phe_list_mode':self.phe_list_mode, 'vec_type':self.vec_type, 'x_sparse':x_sparse,
				'xdtype':np.float32, 'multi_label': c.multi_label, 'use_rd_mix_code': self.use_rd_mix_code,
				'y_one_hot':True, 'ydtype':np.float32, 'true':c.pertur_weight[0], 'reduce':c.pertur_weight[1],
				'rise':c.pertur_weight[2],
				'lower':c.pertur_weight[3], 'noise':c.pertur_weight[4]
			})
			return RandomGenBatchController(bc_config, self.hpo_reader, seed=c.seed)
		dh = DataHelper(self.hpo_reader)
		X = dh.get_train_X(self.phe_list_mode, self.vec_type, sparse=x_sparse, dtype=np.float32)
		y_, y_col_names = dh.get_train_y(one_hot=True, dtype=np.float32, use_rd_mix_code=self.use_rd_mix_code,
			multi_label=c.multi_label, expand_ances=c.dis_expand_ances, desc_dp=c.desc_dp)
		assert y_col_names == self.dis_list
		return BatchControllerMat([X, y_], seed=c.seed)


	def get_batch_controller(self, c):
		if c.mixup:
			print('Using Mixup')
			c1 = deepcopy(c); bc1 = self.get_single_batch_controller(c1)
			c2 = deepcopy(c); c2.seed += 100; bc2 = self.get_single_batch_controller(c2)
			return BatchControllerMixupMat(c.mix_alpha, bc1, bc2, seed=c.seed)
		else:
			return self.get_single_batch_controller(c)


	def gen_val_test_data_dict(self, data_names, part, c):
		"""
		Returns:
			dict: {data_name: (X, y_)}
		"""
		print('get test data...')
		d = {}
		dh = DataHelper(self.hpo_reader)
		for name in data_names:
			d[name] = dh.get_test_val_Xy(name, part, self.phe_list_mode, self.vec_type, x_sparse=True, xdtype=np.float32,
				y_one_hot=True, ydtype=np.float32, use_rd_mix_code=self.use_rd_mix_code)
		return d


	def gen_val_test_feed_dict(self, data_dict):
		"""
		Args:
			data_dict (dict): {data_name: (X, y_)}
		Returns:
			dict: {data_name: feed_dict}
		"""
		ret_dict = {}
		for data_name, (X, (y_, y_col_names)) in data_dict.items():
			assert y_col_names == self.dis_list
			X, x_val_shape = self.sparse_X_to_input(X)
			ret_dict[data_name] = {
				self.placeholders['X']:X,
				self.placeholders['x_val_shape']:x_val_shape,
				self.placeholders['y_']:sparse_to_tuple(y_),
				self.placeholders['keep_prob']:1.0,
				self.placeholders['seqy_']:y_.argmax(axis=1).A.astype(np.int64)
			}
		return ret_dict


	def get_loss(self, logits, c):
		return self.get_clt_loss(logits, c) + self.get_l2_loss(c)


	def get_clt_loss(self, logits, c):
		y_ = tf.sparse.to_dense(self.placeholders['y_'])
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))


	def get_l2_loss(self, c):

		return tf.contrib.layers.apply_regularization(
			tf.contrib.layers.l2_regularizer(c.w_decay), tf.trainable_variables())


	def gen_summary(self, c):
		if not c.summary:
			return
		self.summary_dict['train_loss'] = tf.summary.scalar('train_loss', self.loss)
		for data_name in c.val_data_names:
			summ_op_list = [tf.summary.scalar('test_loss/{}'.format(data_name), self.loss)]
			for k in c.test_acc_ats:
				summ_op_list.append(tf.summary.scalar('test_recall_at{}/{}'.format(k, data_name), self.metric_dict['recall@{}'.format(k)]))
			self.summary_dict[data_name] = tf.summary.merge(summ_op_list)
		self.summary_dict['weight'] = tf.summary.merge([
			tf.summary.histogram('W', self.vars['W']),
			tf.summary.histogram('b', self.vars['b']),
			tf.summary.histogram('logits', self.logits),
		])


	def gen_metric(self, c):
		for k in c.test_acc_ats:
			_, self.metric_dict['recall@{}'.format(k)] = tf.metrics.recall_at_k(self.placeholders['seqy_'], self.prob, k)


	@timer
	def build(self, c):
		if c.seed is not None:
			print('Set Random Seed: {}'.format(c.seed))
			tf.set_random_seed(c.seed)
		self.gen_placeholders(c)
		X = self.placeholders['X']
		X = dropout(X, self.placeholders['keep_prob'], self.placeholders['x_val_shape'])
		W = glorot((c.n_features, c.class_num), 'W'); self.vars['W'] = W
		b = zeros((c.class_num,), 'b'); self.vars['b'] = b
		self.logits = dot(X, W, sparse=(type(X)==tf.SparseTensor)) + b
		self.output_logits = tf.cast(self.logits, tf.float64)
		self.prob = tf.nn.sigmoid(self.output_logits) #
		self.gen_metric(c)

		self.loss = self.get_loss(self.logits, c)
		self.global_step = tf.Variable(0, trainable=False); self.vars['global_step'] = self.global_step
		self.train_op = get_optimizer(c.optimizer, c.lr).minimize(self.loss, global_step=self.global_step)
		self.gen_summary(c)
		self.init_op = tf.global_variables_initializer()
		self.local_init_op = tf.local_variables_initializer()


	def _train(self, c, from_last=False):
		if from_last:
			c.load(self.CONFIG_JSON)
		logger = get_logger(self.name, self.LOG_PATH, mode='w')
		logger.info(self.name)
		logger.info(c)
		bc = self.get_batch_controller(c)
		self.build(c)
		val_feed_dict = self.gen_val_test_feed_dict(self.gen_val_test_data_dict(c.val_data_names, c.eval_data, c))

		# FIXME
		mt_val = ModelTestor(c.eval_data, hpo_reader=self.hpo_reader)
		mt_val.load_test_data(c.mt_val_data_names); mt_val_metric_set = set(c.mt_val_metrics)
		mt_test = ModelTestor(TEST_DATA, hpo_reader=self.hpo_reader)
		mt_test.load_test_data(c.mt_test_data_names); mt_test_metric_set = set(c.mt_test_metrics)

		max_total_step = c.max_epoch_num * self.DIS_CODE_NUMBER // c.batch_size
		min_total_step = c.min_epoch_num * self.DIS_CODE_NUMBER // c.batch_size
		logger.info('Total Step: {}; Min Step: {}'.format(max_total_step, min_total_step))
		saver = tf.train.Saver()
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		sess = tf.Session(config=sess_config, graph=self.g)
		self.set_session(sess)
		if from_last:
			saver.restore(sess, self.MODEL_PATH)
			self.load_val_history()
			best_his_rank = self.history_helper.get_arg_best(sort_type=c.hyper_score_type)
			best_gs = self.history_helper.get_H(best_his_rank)['PARAMETER']['gs']
		else:
			self.delete_model(); os.makedirs(self.SAVE_FOLDER, exist_ok=True)
			self.init_history_recorder(c)
			sess.run(self.init_op)
			self.save(c, sess, saver, logger)
			best_his_rank = 0; best_gs = 0



		val_model = OrderedMultiModel(
			hpo_reader=self.hpo_reader, model_list=[self, RandomModel(hpo_reader=self.hpo_reader, seed=777)],
			keep_raw_score=False, model_name=self.name+'-Random')

		min_val_loss = np.inf; no_improve_num = 0
		summary_writer = tf.summary.FileWriter(self.SUMMARY_FOLDER, sess.graph, flush_secs=30) if c.summary else None
		for i in range(1, max_total_step):
			feed_dict = self.gen_feed_dict(c, bc)
			_, loss, gs = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict, options=tf.RunOptions(report_tensor_allocations_upon_oom = True))

			if i % c.print_freq == 0:
				logger.info('Epoch {}({:.4}%); Global Step {}: Batch Loss={}'.format(i, 100*i/max_total_step, gs, loss))
			if c.summary and i % c.summary_freq == 0:
				logger.info('Epoch {}({:.4}%); Global Step {}: Add Summary'.format(i, 100*i/max_total_step, gs))
				summary_writer.add_summary(sess.run(self.summary_dict['train_loss'], feed_dict=feed_dict), gs)
				summary_writer.add_summary(sess.run(self.summary_dict['weight'], feed_dict=feed_dict), gs)
				for data_name in val_feed_dict:
					sess.run(self.local_init_op)
					summary_writer.add_summary(sess.run(self.summary_dict[data_name], feed_dict=val_feed_dict[data_name]), gs)


			if i % c.mt_test_freq == 0 or i == max_total_step-1:

				all_val_loss = {dname:sess.run(self.loss, feed_dict) for dname, feed_dict in val_feed_dict.items()}
				val_loss = np.mean(list(all_val_loss.values()))
				self.add_loss(gs, loss, 'train')
				self.add_loss(gs, val_loss, 'validation')

				mt_metric_dict = mt_val.cal_metric_for_multi_data(val_model, data_names=c.mt_val_data_names,
					metric_set=mt_val_metric_set, use_query_many=True, logger=logger, cpu_use=1)
				for dname, metric_dict in mt_metric_dict.items():
					if 'Mic.RankMedian' in metric_dict:
						metric_dict['Mic.RankMedian'] = -metric_dict['Mic.RankMedian']
				self.add_val_history(int(gs), mt_metric_dict)
				self.save_val_history()

				history = self.history_helper.get_enrich_history(sort_type=c.hyper_score_type)
				best_score, cur_score = history[best_his_rank]['SCORE'], history[-1]['SCORE']

				if cur_score > best_score:
					self.save(c, sess, saver, logger)
					best_his_rank = self.history_helper.get_history_length() - 1; best_gs = gs
					best_score = cur_score
					no_improve_num = 0
				else:
					no_improve_num += 1
				logger.info('Step {}({:.4}%); Global Step {}; Current Score = {}; Best Score = {}, Best Step = {}'.format(
						i, 100 * i / max_total_step, gs, cur_score, best_score, best_gs))

				if c.early_stop_patience is not None and gs > min_total_step and no_improve_num > c.early_stop_patience:
					logger.info('Early Stop: Epoch {}({:.4}%); Global Step {}; Best Global Step = {}, Best Score = {}'.format(i, 100*i/max_total_step, gs, best_gs, best_score))
					break

			if i % c.mt_draw_save_freq == 0 or i == max_total_step-1:
				self.draw_history_loss()
				self.draw_val_history(c.mt_val_data_names, c.mt_draw_metrics, c.hyper_score_type)

		delete_logger(logger)
		sess.close()


	def init_history_recorder(self, c):
		self.history_helper = HyperTuneHelper(
			None,
			score_keys=[c.mt_val_data_names, c.mt_val_metrics],
			score_weights=c.val_rank_weights,
			mode='a',
			save_folder=self.VAL_HISTORY_FOLDER
		)
		self.history_loss = {
			'step_list': [],
			'loss_list': [],
			'val_step_list': [],
			'val_loss_list': []
		}


	def load_val_history(self):
		if os.path.exists(self.VAL_HISTORY_FOLDER):
			self.history_helper = HyperTuneHelper(None, mode='a', save_folder=self.VAL_HISTORY_FOLDER)
			json.load(open(self.HISTORY_LOSS_JSON))
		else:
			assert False


	def save_val_history(self):
		self.history_helper.save_history()
		json.dump(self.history_loss, open(self.HISTORY_LOSS_JSON, 'w'), indent=2)


	def add_loss(self, global_step, loss, loss_type):
		"""
		Args:
			loss_type (str): 'train' | 'validation'
		"""
		if loss_type == 'train':
			self.history_loss['step_list'].append(int(global_step))
			self.history_loss['loss_list'].append(float(loss))
		elif loss_type == 'validation':
			self.history_loss['val_step_list'].append(int(global_step))
			self.history_loss['val_loss_list'].append(float(loss))
		else:
			raise RuntimeError('Unknown loss type: {}'.format(loss_type))


	def add_val_history(self, gs, dname_to_metric):
		self.history_helper.add({'gs': gs}, score_dict=dname_to_metric)


	def draw_val_history(self, draw_data_names, draw_metrics, hyper_score_type):
		print('drawing...')
		history = self.history_helper.get_enrich_history(hyper_score_type)
		gs_list = [h['PARAMETER']['gs'] for h in history]
		line_names, y_list = [], []
		for dt in draw_data_names:
			for me in draw_metrics:
				line_names.append(dt+'_'+me)
				score_key = self.history_helper.get_score_key([dt, me])
				y_list.append([h['FLT_SCORE_DICT'][score_key] for h in history])
		line_names.append(hyper_score_type)
		is_score_list = isinstance(history[0]['SCORE'], list) or isinstance(history[0]['SCORE'], tuple)
		y_list.append([ (h['SCORE'][0] if is_score_list else h['SCORE']) for h in history])
		x_list = [gs_list] * len(y_list)
		simple_multi_line_plot(self.PERFORMANCE_FIG_PATH, x_list, y_list, line_names, 'Step', 'Score', 'Performance Plot', figsize=(20, 20))


	def draw_history_loss(self):
		simple_multi_line_plot(
			self.HISTORY_LOSS_FIG_PATH,
			[self.history_loss['step_list'], self.history_loss['val_step_list']],
			[self.history_loss['loss_list'], self.history_loss['val_loss_list']],
			line_names=['train_loss', 'val_loss'],
			x_label='Step', y_label='Loss',
			title='Loss Plot', figsize=(20, 20))


	def train(self, c, save_model=True, from_last=False):
		"""
		Args:
			save_model (bool): ignorre
		"""
		tf.reset_default_graph()
		self.g = tf.Graph()
		with self.g.as_default():
			self._train(c, from_last)


	def save(self, c, sess=None, saver=None, logger=None):
		sess = sess or self.sess
		saver = saver or tf.train.Saver()
		print('save model:', self.MODEL_PATH)
		path = saver.save(sess, self.MODEL_PATH)
		c.save(self.CONFIG_JSON)
		self.save_val_history()
		if logger is not None:
			logger.info('Model saved in path: {}'.format(path))


	def restore(self, config_initializer=LRNeuronConfig):
		self.g = tf.Graph()
		tf.reset_default_graph()
		c = config_initializer(); c.load(self.CONFIG_JSON)
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		sess = tf.Session(config=sess_config, graph=self.g)
		with self.g.as_default():
			self.build(c)
			saver = tf.train.Saver()
			saver.restore(sess, self.MODEL_PATH)
			self.set_session(sess)


	def change_save_folder_and_save(self, model_name=None, save_folder=None):
		old_model_path = self.MODEL_PATH
		old_save_folder = self.SAVE_FOLDER
		self.name = model_name or self.name
		self.init_save_folder(save_folder)
		shutil.rmtree(self.SAVE_FOLDER, ignore_errors=True)
		os.system('cp -r {} {}'.format(old_save_folder, self.SAVE_FOLDER))
		checkpoint = self.SAVE_FOLDER + '/checkpoint'
		content = open(checkpoint).read()
		open(checkpoint, 'w').write(content.replace(old_model_path, self.MODEL_PATH))



	def delete_model(self):
		shutil.rmtree(self.SAVE_FOLDER, ignore_errors=True)


	def set_session(self, sess):
		self.sess = sess


	def predict_prob(self, X):

		feed_dict = {self.placeholders['keep_prob']: 1.0}
		if sp.issparse(X):
			X = sparse_to_tuple(X)
			feed_dict[self.placeholders['x_val_shape']] = X[1].shape
		feed_dict[self.placeholders['X']] = X
		return self.sess.run(self.output_logits, feed_dict=feed_dict)


if __name__ == '__main__':
	model = LRNeuronModel(model_name='LRNeuTestModel')
	print(model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463']))



