
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json

from bert_syn.bert_pkg import BertConfig, BertModel, optimization, modeling, tokenization
from bert_syn.core.model_config import Config
from bert_syn.core.sim_base import SimBase, SimBaseModel
from bert_syn.core.input import CsvDataProcessor, MemDataProcessor
from bert_syn.utils.constant import DATA_PATH, MODEL_PATH
from bert_syn.utils.utils import check_return, timer
from bert_syn.utils.utils_draw import simple_multi_line_plot


class BertSimConfig(Config):
	def __init__(self):
		super(BertSimConfig, self).__init__()

		# # for bert
		# self.bert_init_model_path = os.path.join(MODEL_PATH, 'bert', 'chinese_L-12_H-768_A-12')
		# self.init_checkpoint = os.path.join(self.bert_init_model_path, 'bert_model.ckpt')
		# self.bert_config_path = os.path.join(self.bert_init_model_path, 'bert_config.json')
		# self.vocab_file = os.path.join(self.bert_init_model_path, 'vocab.txt')
		# self.train_batch_size = 128
		# self.eval_batch_size = 128
		# self.predict_batch_size = 128
		# self.num_train_epochs = 5

		# for albert-base
		# self.bert_init_model_path = os.path.join(MODEL_PATH, 'albert_google', 'albert_base')
		# self.init_checkpoint = os.path.join(self.bert_init_model_path, 'model.ckpt')
		# self.bert_config_path = os.path.join(self.bert_init_model_path, 'albert_config.json')
		# self.vocab_file = os.path.join(self.bert_init_model_path, 'vocab_chinese.txt')
		# self.train_batch_size = 256
		# self.eval_batch_size = 256
		# self.predict_batch_size = 256
		# self.num_train_epochs = 3

		# for albert_zh-tiny (brightmart)
		self.bert_init_model_path = os.path.join(MODEL_PATH, 'albert_brightmart', 'albert_tiny_zh_google')
		self.init_checkpoint = os.path.join(self.bert_init_model_path, 'albert_model.ckpt')
		self.bert_config_path = os.path.join(self.bert_init_model_path, 'albert_config_tiny_g.json')

		self.vocab_file = os.path.join(self.bert_init_model_path, 'vocab.txt')
		self.train_batch_size = 1024
		self.eval_batch_size = 1024
		self.predict_batch_size = 1024
		self.num_train_epochs = 5


		self.train_data_path = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'mxh', 'train.csv')  # set to None if no predict

		self.eval_data_path = None

		# self.predict_data_path = None
		self.predict_data_path = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'test', 'test_all.csv')  # set to None if no predict
		self.predict_raw_to_true_json = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'test', 'ehr_to_true_texts_all.json')

		self.init_pretrain = True
		self.do_lower_case = True
		# self.max_seq_length = 64
		self.max_seq_length = 32
		self.learning_rate = 5e-5
		self.warmup_proportion = 0.1
		self.shuffle_buffer_size = 10000

		self.print_freq = 10
		self.eval_freq = 1000
		self.predict_freq = 1000
		self.save_freq = 1000
		self.draw_freq = 200


class BertSimModel(SimBaseModel):
	def __init__(self, mode, config):
		"""
		Args:
			mode (str): tf.estimator.ModeKeys.TRAIN | tf.estimator.ModeKeys.EVAL | tf.estimator.ModeKeys.PREDICT
			config (BertSimConfig):
		"""
		super(BertSimModel, self).__init__()
		self.mode = mode
		self.config = config
		self.bert_config = BertConfig.from_json_file(config.bert_config_path)


	def forward(self, data, num_labels, num_train_steps=None):
		input_ids = data["input_ids"]
		input_mask = data["input_mask"]
		segment_ids = data["segment_ids"]
		label_ids = data["label_ids"]

		is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)

		model = BertModel(config=self.bert_config, is_training=is_training, input_ids=input_ids,
			input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=False)    # bert

		# In the demo, we are doing a simple classification task on the entire
		# segment.
		#
		# If you want to use the token-level output, use model.get_sequence_output()
		# instead.
		output_layer = model.get_pooled_output()

		hidden_size = output_layer.shape[-1].value

		output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

		output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

		if is_training:
			# I.e., 0.1 dropout
			output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

		self.global_step = tf.train.get_or_create_global_step()

		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		self.probs = tf.nn.softmax(tf.cast(logits, tf.float64), axis=-1)[:, 1]
		self.y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
		self.y_true = label_ids

		if self.mode == tf.estimator.ModeKeys.PREDICT:
			return

		one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		self.per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		self.loss = tf.reduce_mean(self.per_example_loss)

		if self.mode == tf.estimator.ModeKeys.EVAL:
			return

		assert num_train_steps is not None
		num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)
		self.train_op = optimization.create_optimizer(
			self.loss, self.config.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
		if self.config.init_pretrain:
			self.init_bert_pretrain(self.config)
		self.init_op = tf.global_variables_initializer()


# ====================================================================
class BertSim(SimBase):
	def __init__(self, name=None, save_folder=None, config=None, user_mode=False):
		super(BertSim, self).__init__(name, save_folder)
		self.config = config or BertSimConfig()
		self.user_mode = user_mode
		self.tokenizer = None
		self.processor = None
		self.mem_processor = None
		self.label_list = None
		self.pred_examples = None


	def init_path(self, save_folder):
		super(BertSim, self).init_path(save_folder)
		self.HISTORY_EVAL_ACC_FIG_PATH = os.path.join(self.SAVE_FOLDER, 'eval_acc.png')


	def build(self):
		config = self.config
		num_labels = len(self.get_label_list())

		if self.user_mode:
			with tf.name_scope('UserPredict'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.user_pred_data, self.user_pred_data_init_op, self.user_pred_data_ph = self.get_ph_dataset(
						config.max_seq_length, config.predict_batch_size, False, False)
					self.user_pred_model = BertSimModel(tf.estimator.ModeKeys.PREDICT, config)
					self.user_pred_model.forward(self.user_pred_data, num_labels)
			return

		if config.train_data_path:
			with tf.name_scope('Train'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.train_data, self.train_data_init_op, self.train_data_info = self.get_train_data(config)
					self.num_train_steps = int(self.train_data_info['n_samples'] / config.train_batch_size * config.num_train_epochs)
					self.train_model = BertSimModel(tf.estimator.ModeKeys.TRAIN, config)
					self.train_model.forward(self.train_data, num_labels, self.num_train_steps)
		if config.eval_data_path:
			with tf.name_scope('Eval'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.eval_data, self.eval_data_init_op, self.eval_data_info = self.get_eval_data(config)
					self.eval_model = BertSimModel(tf.estimator.ModeKeys.EVAL, config)
					self.eval_model.forward(self.eval_data, num_labels)
		if config.predict_data_path:
			with tf.name_scope('Predict'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.pred_data, self.pred_data_init_op, self.pred_data_info = self.get_predict_data(config)
					self.pred_model = BertSimModel(tf.estimator.ModeKeys.PREDICT, config)
					self.pred_model.forward(self.pred_data, num_labels)


	def train(self, from_last=False):
		self.g = tf.Graph()
		with self.g.as_default():
			return self._train(from_last)


	def _train(self, from_last=False):
		sess = self.get_sess()
		if from_last:
			self.config.load(self.CONFIG_JSON)
			self.build()
			saver = tf.train.Saver(max_to_keep=100)
			tf.logging.info('Loading from last...')
			saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(self.MODEL_PATH)))
			self.load_history()
		else:
			self.delete_model(); os.makedirs(self.SAVE_FOLDER)
			self.build()
			saver = tf.train.Saver(max_to_keep=100)
			self.init_history()
			sess.run(self.train_model.init_op)
		self.train_model.print_trainable_vars()
		print(self.config)
		sess.run(self.train_data_init_op)
		global_step = sess.run(self.train_model.global_step)
		while global_step <= self.num_train_steps:
			sess.run(self.train_model.train_op)
			global_step = sess.run(self.train_model.global_step)
			if global_step != 0 and global_step % self.config.print_freq == 0:
				loss = sess.run(self.train_model.loss)
				tf.logging.info('Global step = {}({:.4}%); Batch loss = {}'.format(
					global_step, 100*global_step/self.num_train_steps, loss))
				self.add_historys({'step': global_step, 'loss': loss})
			if self.config.eval_data_path is not None and global_step != 0 and global_step % self.config.eval_freq == 0:
				eval_loss, eval_accuracy = self.eval()
				self.add_historys({'eval_step': global_step, 'eval_loss': eval_loss, 'eval_acc': eval_accuracy})
				tf.logging.info('Global step = {}({:.4}%); Eval loss = {}; Eval accuracy = {}'.format(
					global_step, 100*global_step/self.num_train_steps, eval_loss, eval_accuracy))
			if self.config.predict_data_path is not None and global_step != 0 and global_step % self.config.predict_freq == 0:
				self.predict()
			if global_step != 0 and global_step % self.config.draw_freq == 0:
				self.draw_history()
			if global_step != 0 and global_step % self.config.save_freq == 0:
				self.save(saver, global_step=global_step)
		# self.predict()
		self.draw_history()
		self.save(saver)


	@timer
	def eval(self):
		sess = self.get_sess()
		sess.run(self.eval_data_init_op)
		loss_list, y_pred, y_true = [], [], []
		with tqdm(total=self.eval_data_info['n_samples']) as pbar:
			try:
				while True:
					per_example_loss, batch_y_pred, batch_y_true = sess.run(
						[self.eval_model.per_example_loss, self.eval_model.y_pred, self.eval_model.y_true])
					loss_list.append(per_example_loss); y_pred.append(batch_y_pred); y_true.append(batch_y_true)
					pbar.update(len(per_example_loss))
			except tf.errors.OutOfRangeError:
				pass
		loss = np.mean(np.hstack(loss_list))
		accuracy = accuracy_score(np.hstack(y_true), np.hstack(y_pred))
		return loss, accuracy


	@timer
	def predict(self):
		sess = self.get_sess()
		sess.run(self.pred_data_init_op)
		global_step = sess.run(self.pred_model.global_step)
		probs = []
		with tqdm(total=self.pred_data_info['n_samples']) as pbar:
			try:
				while True:
					batch_prob = sess.run(self.pred_model.probs)
					probs.append(batch_prob)
					pbar.update(len(batch_prob))
			except tf.errors.OutOfRangeError:
				pass
		probs = np.hstack(probs)
		pred_examples = self.get_pred_examples()
		for e, p in zip(pred_examples, probs):
			e.label = p
		self.save_and_cal_metric_for_examples(
			global_step, pred_examples, self.get_processor(),
			json.load(open(self.config.predict_raw_to_true_json)))


	def predict_probs(self, sent_pairs):
		"""
		Args:
			sent_pairts (list): [(str1, str2), ...]
		Returns:
			np.ndarray: shape=(len(sent_pairs),); dtype=np.float64
		"""
		examples = self.get_mem_processor().get_examples(sent_pairs, 'pred')
		feed_dict = self.get_processor().convert_examples_to_feed_dict(examples, self.get_label_list(),
			self.config.max_seq_length, self.get_tokenizer())
		sess = self.get_sess()
		sess.run(self.user_pred_data_init_op, feed_dict={self.user_pred_data_ph[k]: feed_dict[k] for k in self.user_pred_data_ph})
		probs = []
		with tqdm(total=len(sent_pairs)) as pbar:
			try:
				while True:
					batch_probs = sess.run(self.user_pred_model.probs)
					probs.append(batch_probs)
					pbar.update(len(batch_probs))
			except tf.errors.OutOfRangeError:
				pass
		return np.hstack(probs)


	def predict_prob(self, sent1, sent2):
		"""
		Args:
			sent1 (str)
			sent2 (str)
		Returns:
			float
		"""
		return float(self.predict_probs([(sent1, sent2)])[0])


	def get_ph_dataset(self, max_seq_length, batch_size, is_training, drop_remainder):
		"""
		Returns:
			tf.Dataset
			dict: {placeholder_name: tf.placeholder}
		"""
		placeholder_dict = {
			'input_ids': tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids'),
			'input_mask': tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_mask'),
			'segment_ids': tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids'),
			'label_ids': tf.placeholder(tf.int32, shape=[None], name='label_ids'),
		}
		ds = tf.data.Dataset.from_tensor_slices(placeholder_dict)
		if is_training:
			ds = ds.repeat()
			ds = ds.shuffle(buffer_size=self.config.shuffle_buffer_size)
		ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, placeholder_dict


	def get_tfrecord_dataset(self, data_path, guid_mark, max_seq_length, batch_size, is_training, drop_remainder):
		processor = self.get_processor()
		tfrecord_path, data_info = processor.get_tfrecord(data_path, guid_mark, max_seq_length)
		ds = processor.tfrecord_to_dataset(tfrecord_path, max_seq_length, is_training=is_training, drop_remainder=drop_remainder,
			batch_size=batch_size, buffer_size=self.config.shuffle_buffer_size)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, data_info


	def get_train_data(self, config):
		return self.get_tfrecord_dataset(config.train_data_path, 'train',
			config.max_seq_length, config.train_batch_size, is_training=True, drop_remainder=True)


	def get_eval_data(self, config):
		return self.get_tfrecord_dataset(config.eval_data_path, 'eval',
			config.max_seq_length, config.eval_batch_size, is_training=False, drop_remainder=False)


	def get_predict_data(self, config):
		return self.get_tfrecord_dataset(config.predict_data_path, 'pred',
			config.max_seq_length, config.predict_batch_size, is_training=False, drop_remainder=False)


	def init_bert_pretrain(self, config, **kwargs):
		super(BertSim, self).init_bert_pretrain(config, modeling)


	@check_return('tokenizer')
	def get_tokenizer(self):
		return tokenization.FullTokenizer(vocab_file=self.config.vocab_file, do_lower_case=self.config.do_lower_case)


	@check_return('processor')
	def get_processor(self):
		return CsvDataProcessor(self.get_tokenizer())


	@check_return('mem_processor')
	def get_mem_processor(self):
		return MemDataProcessor(self.get_tokenizer())


	@check_return('label_list')
	def get_label_list(self):
		return self.get_processor().get_labels()


	@check_return('pred_examples')
	def get_pred_examples(self):
		return self.get_processor().get_examples(self.config.predict_data_path, 'pred')


	def draw_history(self):
		super(BertSim, self).draw_history()
		simple_multi_line_plot(
			self.HISTORY_EVAL_ACC_FIG_PATH,
			[self.history['eval_step']], [self.history['eval_acc']],
			line_names=['Eval Acc'], x_label='Step', y_label='Accuracy', title='Eval Accuracy'
		)

if __name__ == '__main__':
	pass