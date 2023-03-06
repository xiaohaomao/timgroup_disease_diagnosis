

import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import heapq

from bert_syn.bert_pkg import BertConfig, BertModel, optimization, modeling, tokenization
from bert_syn.core.model_config import Config
from bert_syn.core.sim_base import SimBase, SimBaseModel
from bert_syn.core.input_ddml import CsvDataProcessor, MemDataProcessor, InputExample
from bert_syn.utils.constant import DATA_PATH, MODEL_PATH, RESULT_PATH, TANH
from bert_syn.utils.utils import check_return, timer, euclidean_distances
from bert_syn.utils.utils_tf import get_active_func, euclidean_dist2, euclidean_dist


class BertDDMLConfig(Config):
	def __init__(self):
		super(BertDDMLConfig, self).__init__()

		# for bert
		# self.bert_init_model_path = os.path.join(MODEL_PATH, 'bert', 'chinese_L-12_H-768_A-12')
		# self.init_checkpoint = os.path.join(self.bert_init_model_path, 'bert_model.ckpt')
		# self.bert_config_path = os.path.join(self.bert_init_model_path, 'bert_config.json')
		# self.vocab_file = os.path.join(self.bert_init_model_path, 'vocab.txt')
		# self.train_batch_size = 64
		# self.eval_batch_size = 64
		# self.predict_batch_size = 64
		# self.num_train_epochs = 5

		# for albert_zh-tiny (brightmart)
		self.bert_init_model_path = os.path.join(MODEL_PATH, 'albert_brightmart', 'albert_tiny_zh_google')
		self.init_checkpoint = os.path.join(self.bert_init_model_path, 'albert_model.ckpt')
		self.bert_config_path = os.path.join(self.bert_init_model_path, 'albert_config_tiny_g.json')
		self.vocab_file = os.path.join(self.bert_init_model_path, 'vocab.txt')
		self.train_batch_size = 1024
		self.eval_batch_size = 1024
		self.predict_batch_size = 1024
		self.num_train_epochs = 5

		dataset_path = os.path.join(DATA_PATH, 'preprocess', 'dataset')




		self.train_data_path = os.path.join(dataset_path, "2020_11_12_mxh_old_best_bert_base_800w_data", 'train.csv')

		self.eval_data_path = None

		self.predict_data_path = os.path.join(dataset_path, 'pumc', 'test', 'ehr_terms_all.csv')  # set to None if no predict
		self.predict_raw_to_true_json = os.path.join(dataset_path, 'pumc', 'test', 'ehr_to_true_texts_all.json')

		self.init_pretrain = True
		self.do_lower_case = True
		self.max_seq_length = 16


		self.learning_rate = 5e-5
		self.warmup_proportion = 0.1
		self.shuffle_buffer_size = 10000

		self.print_freq = 10
		self.eval_freq = 1000
		self.predict_freq = 500
		self.save_freq = 500
		self.draw_freq = 1000

		self.fc_layers = [(1024, None)]
		self.tau = 2.0


class BertDDMLModel(SimBaseModel):
	def __init__(self, mode, config):
		"""
		Args:
			mode (str): tf.estimator.ModeKeys.TRAIN | tf.estimator.ModeKeys.EVAL | tf.estimator.ModeKeys.PREDICT
			config (BertDDMLConfig):
		"""
		super(BertDDMLModel, self).__init__()
		self.mode = mode
		self.config = config
		self.bert_config = BertConfig.from_json_file(config.bert_config_path)


	def forward(self, data, num_train_steps=None):
		input_ids = data["input_ids"]
		input_mask = data["input_mask"]
		segment_ids = data["segment_ids"]
		label = data["label"]

		is_training = (self.mode == tf.estimator.ModeKeys.TRAIN)
		self.global_step = tf.train.get_or_create_global_step()

		model = BertModel(config=self.bert_config, is_training=is_training, input_ids=input_ids,
			input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=False)  # bert

		output_layer = model.get_pooled_output()
		if is_training:
			output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
		for i, (unit, act) in enumerate(self.config.fc_layers, 1):
			output_layer = tf.layers.dense(output_layer, unit,
				activation=get_active_func(act),
				kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
				name=f'fc_layer_{i}',
			)
			if is_training:
				output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
		self.embedding = output_layer

		if self.mode == tf.estimator.ModeKeys.PREDICT:
			return

		pair_num = tf.cast(tf.shape(self.embedding)[0] / 2, tf.int32)
		embed_a, embed_b = self.embedding[:pair_num], self.embedding[pair_num:]
		dist2 = euclidean_dist2(embed_a, embed_b)   # (batch_size,)
		# dist2 = euclidean_dist(embed_a, embed_b, 1e-6)  # (batch_size,)
		self.per_example_loss = tf.nn.relu(1. - label * (self.config.tau - dist2))
		self.loss = tf.reduce_mean(self.per_example_loss)

		if self.mode == tf.estimator.ModeKeys.EVAL:
			return

		assert num_train_steps is not None
		num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)
		self.train_op = optimization.create_optimizer(
			self.loss, self.config.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)


		if hasattr(self.config, 'init_pretrain') and self.config.init_pretrain:
			self.init_bert_pretrain(self.config)
		self.init_op = tf.global_variables_initializer()


class BertDDMLSim(SimBase):
	def __init__(self, name=None, save_folder=None, config=None, user_mode=False):
		super(BertDDMLSim, self).__init__(name or 'BertDDMLSim', save_folder)
		self.config = config or BertDDMLConfig()
		self.user_mode = user_mode

		self.dict_terms = None
		self.dict_terms_feed_dict = None
		self.dict_embedding = None




	def build(self):
		config = self.config



		with tf.name_scope('UserPredict'):
			with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
				self.user_pred_data, self.user_pred_data_init_op, self.user_pred_data_ph = self.get_ph_dataset(
					config.max_seq_length, config.predict_batch_size, False, False)


				self.user_pred_model = BertDDMLModel(tf.estimator.ModeKeys.PREDICT, config)
				self.user_pred_model.forward(self.user_pred_data)

		if config.train_data_path:
			with tf.name_scope('Train'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.train_data, self.train_data_init_op, self.train_data_info = self.get_train_data(config)
					self.num_train_steps = int(self.train_data_info['n_samples'] / config.train_batch_size * config.num_train_epochs)
					self.train_model = BertDDMLModel(tf.estimator.ModeKeys.TRAIN, config)
					self.train_model.forward(self.train_data, self.num_train_steps)
		if config.eval_data_path:
			with tf.name_scope('Eval'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.eval_data, self.eval_data_init_op, self.eval_data_info = self.get_eval_data(config)
					self.eval_model = BertDDMLModel(tf.estimator.ModeKeys.EVAL, config)
					self.eval_model.forward(self.eval_data)
		if config.predict_data_path:
			with tf.name_scope('Predict'):
				with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
					self.pred_data, self.pred_data_init_op, self.pred_data_info = self.get_predict_data(config, drop_text_b=True)
					self.pred_model = BertDDMLModel(tf.estimator.ModeKeys.PREDICT, config)
					self.pred_model.forward(self.pred_data)


	def set_dict_terms(self, dict_terms):
		"""
		Args:
			dict_terms (list): [str1, str2, ...]
		"""
		self.dict_terms = dict_terms
		self.dict_terms_feed_dict = self.terms_to_feed_dict(dict_terms, 'dict_term')
		assert self.dict_terms_feed_dict['input_ids'].shape == (len(dict_terms), self.config.max_seq_length)


	def terms_to_feed_dict(self, terms, guid_mark):
		"""
		Args:
			terms (list): [str1, str2, ...]
		Returns:
			dict: {
				'input_ids': [input_ids1, input_ids2, ...],
				'input_mask': [input_mask1, input_mask2],
				'segment_ids': [segment_ids1, segment_ids2, ...]
				'label': [label1, label2, ...]
			}
		"""
		examples = self.get_mem_processor().get_examples([(term, '', -1.) for term in terms], guid_mark=guid_mark)
		return self.get_processor().convert_examples_to_feed_dict(
			examples, self.get_label_list(), self.config.max_seq_length, self.get_tokenizer(), ignore_text_b=True)


	def get_embedding(self, terms=None, feed_dict=None, guid_mark='user_pred'):
		if feed_dict is None:
			feed_dict = self.terms_to_feed_dict(terms, guid_mark)
		sess = self.get_sess()
		sess.run(self.user_pred_data_init_op, feed_dict={self.user_pred_data_ph[k]:feed_dict[k] for k in self.user_pred_data_ph})
		term_embeddings = []
		with tqdm(total=len(feed_dict['input_ids'])) as pbar:
			try:
				while True:
					batch_embedding = sess.run(self.user_pred_model.embedding)
					term_embeddings.append(batch_embedding)
					pbar.update(len(batch_embedding))
			except tf.errors.OutOfRangeError:
				pass
		return np.vstack(term_embeddings)


	def get_dict_embedding(self, update=False):
		if update or self.dict_embedding is None:
			self.dict_embedding = self.get_embedding(feed_dict=self.dict_terms_feed_dict)
		return self.dict_embedding

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
		# while global_step <= 20000:
			sess.run(self.train_model.train_op)
			global_step = sess.run(self.train_model.global_step)
			if global_step != 0 and global_step % self.config.print_freq == 0:
				loss = sess.run(self.train_model.loss)
				tf.logging.info('Global step = {}({:.4}%); Batch loss = {}'.format(
					global_step, 100 * global_step / self.num_train_steps, loss))
				self.add_historys({'step':global_step, 'loss':loss})
			if self.config.eval_data_path is not None and global_step != 0 and global_step % self.config.eval_freq == 0:
				eval_loss = self.eval()
				self.add_historys({'eval_step':global_step, 'eval_loss':eval_loss})
				tf.logging.info('Global step = {}({:.4}%); Eval loss = {}'.format(
					global_step, 100 * global_step / self.num_train_steps, eval_loss))
			if self.config.predict_data_path is not None and global_step != 0 and global_step % self.config.predict_freq == 0:
				self.predict(update_dict_embedding=True)
				terms = [
					'乏力', '皮肤发黑', '泌乳素正常', '患者病情稳定', '水肿减轻', '甲功正常', '营养良好',
					'肾上腺', '胸骨', '小肠', '体重', '身高', '红色',
					'为行', '活动', '精神可', '患者自', '饮食', '术后'
				]
				score_ary, col_names = self.predict_scores(terms)
				for i in range(len(terms)):
					print(terms[i], heapq.nlargest(10, [(score, col_name) for score, col_name in zip(score_ary[i], col_names)]))
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
					per_example_loss = sess.run(self.eval_model.per_example_loss)
					loss_list.append(per_example_loss)
					pbar.update(len(per_example_loss))
			except tf.errors.OutOfRangeError:
				pass
		loss = np.mean(np.hstack(loss_list))
		return loss


	@timer
	def predict(self, update_dict_embedding=False, save_result_csv=False):
		sess = self.get_sess()
		sess.run(self.pred_data_init_op)
		global_step = sess.run(self.pred_model.global_step)
		term_embeddings = []
		with tqdm(total=self.pred_data_info['n_samples']) as pbar:
			try:
				while True:
					batch_embedding = sess.run(self.pred_model.embedding)
					term_embeddings.append(batch_embedding)
					pbar.update(len(batch_embedding))
			except tf.errors.OutOfRangeError:
				pass
		term_embeddings = np.vstack(term_embeddings)
		dict_term_embeddings = self.get_dict_embedding(update_dict_embedding)
		tf.logging.info(f'predict term embedding: {term_embeddings.shape}; dict term embedding: {dict_term_embeddings.shape}')
		scores = -euclidean_distances(term_embeddings, dict_term_embeddings, squared=True)
		pred_examples = self.get_pred_examples()
		examples_to_save = []
		for i in range(len(pred_examples)):
			for j in range(len(self.dict_terms)):
				examples_to_save.append(InputExample(f'pred-{i}-{j}', pred_examples[i].text_a, self.dict_terms[j], scores[i,j]))
		self.save_and_cal_metric_for_examples(
			global_step, examples_to_save, self.get_processor(),
			json.load(open(self.config.predict_raw_to_true_json)),
			save_result_csv=save_result_csv)


	@timer
	def predict_scores(self, terms, update_dict_embedding=False, cpu_use=12):
		"""Note: higher score means better matching
		Args:
			terms (list): [str1, str2, ...]
			update_dict_embedding (bool): whether to update the embedding of terms in dict (set by user)
		Returns:
			np.ndarray: shape=(len(terms), len(dict_terms))
			list: column_names
		"""
		assert self.dict_terms is not None
		term_embeddings = self.get_embedding(terms)
		dict_term_embeddings = self.get_dict_embedding(update_dict_embedding)
		scores = -euclidean_distances(term_embeddings, dict_term_embeddings, cpu_use=cpu_use)
		return scores, self.dict_terms


	@timer
	def predict_best_match(self, terms, update_dict_embedding=False, chunk_size=5000):
		"""
		Args:
			terms (list): [str1, str2, ...]
			update_dict_embedding (bool): whether to update the embedding of terms in dict (set by user)
		Returns:
			list: [(match_str1, score1), (match_str2, score2), ...]; length = len(terms)
		"""
		assert self.dict_terms is not None
		term_embeddings = self.get_embedding(terms)
		dict_term_embeddings = self.get_dict_embedding(update_dict_embedding)
		ret_list = []
		col_names = np.array(self.dict_terms)
		for i in range(0, term_embeddings.shape[0], chunk_size):
			print('Processed: {} / {} ({} %)'.format(i, term_embeddings.shape[0], i * 100. / term_embeddings.shape[0]))
			score_mat = -euclidean_distances(term_embeddings[i:i+chunk_size], dict_term_embeddings)
			idx = score_mat.argmax(axis=1)
			best_scores = score_mat[list(range(score_mat.shape[0])), idx]
			best_terms = col_names[idx]
			ret_list.extend(list(zip(best_terms, best_scores)))
		return ret_list


	def predict_score(self, term, update_dict_embedding=False):
		"""
		Args:
			term (str):
		Returns:
			np.ndarray: shape=(len(dict_terms),)
			list: column_names
		"""
		ary, column_names = self.predict_scores([term], update_dict_embedding)
		return ary.flatten(), column_names


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
			'label': tf.placeholder(tf.float32, shape=[None], name='label'),
		}
		ds = tf.data.Dataset.from_tensor_slices(placeholder_dict)
		if is_training:
			ds = ds.repeat()
			ds = ds.shuffle(buffer_size=self.config.shuffle_buffer_size)
		ds = ds.batch(batch_size=batch_size*2, drop_remainder=drop_remainder)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, placeholder_dict


	def get_tfrecord_dataset(self, data_path, guid_mark, max_seq_length, batch_size, is_training, drop_remainder, drop_text_b=False):
		processor = self.get_processor()
		tfrecord_path, data_info = processor.get_tfrecord(data_path, guid_mark, max_seq_length)
		ds = processor.tfrecord_to_dataset(tfrecord_path, max_seq_length, is_training=is_training, drop_remainder=drop_remainder,
			batch_size=batch_size, buffer_size=self.config.shuffle_buffer_size, drop_text_b=drop_text_b)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, data_info


	def get_train_data(self, config):
		return self.get_tfrecord_dataset(config.train_data_path, 'train',
			config.max_seq_length, config.train_batch_size, is_training=True, drop_remainder=True, drop_text_b=False)


	def get_eval_data(self, config):
		return self.get_tfrecord_dataset(config.eval_data_path, 'eval',
			config.max_seq_length, config.eval_batch_size, is_training=False, drop_remainder=False, drop_text_b=False)


	def get_predict_data(self, config, drop_text_b=False):
		return self.get_tfrecord_dataset(config.predict_data_path, 'pred', config.max_seq_length,
			config.predict_batch_size, is_training=False, drop_remainder=False, drop_text_b=drop_text_b)


	def init_bert_pretrain(self, config, **kwargs):
		super(BertDDMLSim, self).init_bert_pretrain(config, modeling)


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


if __name__ == '__main__':
	pass



