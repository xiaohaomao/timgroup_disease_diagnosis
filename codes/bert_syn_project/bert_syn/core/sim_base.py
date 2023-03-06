


import os
import json
import numpy as np
import tensorflow as tf
import shutil
import re
import collections

from bert_syn.core.model_config import Config
from bert_syn.core.model_testor import ModelTestor
from bert_syn.utils.constant import MODEL_PATH, RESULT_PATH, DATA_PATH
from bert_syn.utils.utils import check_return, timer, split_path
from bert_syn.utils.utils_draw import simple_multi_line_plot
from bert_syn.utils.utils_tf import get_sess_config


class SimBaseModel(object):
	def __init__(self):
		super(SimBaseModel, self).__init__()
		self.initialized_variable_names = {}


	def get_assignment_map_from_checkpoint(self, tvars, init_checkpoint):
		"""Compute the union of the current variables and checkpoint variables."""
		assignment_map = {}
		initialized_variable_names = {}

		name_to_variable = collections.OrderedDict()
		for var in tvars:
			name = var.name
			m = re.match("^(.*):\\d+$", name)
			if m is not None:
				name = m.group(1)
			name_to_variable[name] = var

		scope_name = tf.get_variable_scope().name
		init_vars = tf.train.list_variables(init_checkpoint)
		assignment_map = collections.OrderedDict()
		for x in init_vars:
			(old_name, old_var) = (x[0], x[1])
			new_name = scope_name + '/' + old_name
			if new_name not in name_to_variable:
				continue
			assignment_map[old_name] = new_name
			initialized_variable_names[new_name] = 1
			initialized_variable_names[new_name + ":0"] = 1
		return (assignment_map, initialized_variable_names)


	def init_bert_pretrain(self, config):
		tvars = tf.trainable_variables()
		if config.init_checkpoint:
			(assignment_map, self.initialized_variable_names
			) = self.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
			tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)


	def print_trainable_vars(self):
		tf.logging.info("**** Trainable Variables ****")
		tvars = tf.trainable_variables()
		for var in tvars:
			init_string = ""
			if var.name in self.initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)


class SimBase(object):
	def __init__(self, name=None, save_folder=None):
		self.name = name
		self.init_path(save_folder)
		self.history = None
		self.sess = None




	def init_path(self, save_folder):
		self.SAVE_FOLDER = save_folder or os.path.join(MODEL_PATH, self.name)


		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_PATH = os.path.join(self.SAVE_FOLDER, 'model.ckpt')
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, 'bert_sim_config.json')
		self.LOG_PATH = os.path.join(self.SAVE_FOLDER, 'log')
		self.HISTORY_PATH = os.path.join(self.SAVE_FOLDER, 'history.json')
		self.HISTORY_LOSS_FIG_PATH = os.path.join(self.SAVE_FOLDER, 'loss.png')
		self.RESULT_SAVE_FOLDER = os.path.join(RESULT_PATH, 'model_result', self.name)
		os.makedirs(self.RESULT_SAVE_FOLDER, exist_ok=True)


	def build(self):
		raise NotImplementedError()


	def __del__(self):
		if hasattr(self, 'sess'):
			self.sess.close()


	def save(self, saver, save_model=True, global_step=None):
		if save_model:
			path = saver.save(self.get_sess(), self.MODEL_PATH, global_step=global_step)
			tf.logging.info('Model saved in path: {}'.format(path))
		self.config.save(self.CONFIG_JSON)
		self.save_history()


	def restore(self, config=None, global_step=None):
		self.g = tf.Graph()
		tf.reset_default_graph()
		if config is None:
			self.config = Config()
			self.config.load(self.CONFIG_JSON)
		else:
			self.config = config
		sess = self.get_sess()
		with self.g.as_default():
			self.build()
			saver = tf.train.Saver()
			if global_step is None:
				saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(self.MODEL_PATH)))
			else:
				saver.restore(sess, self.MODEL_PATH+f'-{global_step}')
		self.load_history()


	def init_bert_pretrain(self, config, modeling):
		tvars = tf.trainable_variables()
		if config.init_checkpoint:
			(assignment_map, initialized_variable_names
			) = modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
			tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)


	def init_history(self, keys=None):
		keys = keys or ['step', 'loss', 'eval_step', 'eval_loss', 'eval_acc']
		self.history = {k: [] for k in keys}


	def add_history(self, key, value):
		if key not in self.history:
			self.history[key] = []
		if isinstance(value, np.int32) or isinstance(value, np.int64):
			value = int(value)
		if isinstance(value, np.float32) or isinstance(value, np.float64):
			value = float(value)
		self.history[key].append(value)


	def add_historys(self, info_dict):
		for k, v in info_dict.items():
			self.add_history(k, v)


	def draw_history(self):
		simple_multi_line_plot(
			self.HISTORY_LOSS_FIG_PATH,
			[self.history['step'], self.history['eval_step']],
			[self.history['loss'], self.history['eval_loss']],
			line_names=['train_loss', 'eval_loss'],
			x_label='Step', y_label='Loss',
			title='Loss Plot')


	def save_history(self):
		json.dump(self.history, open(self.HISTORY_PATH, 'w'))


	def load_history(self):
		self.history = json.load(open(self.HISTORY_PATH))


	def delete_model(self):
		shutil.rmtree(self.SAVE_FOLDER, ignore_errors=True)
		shutil.rmtree(self.RESULT_SAVE_FOLDER, ignore_errors=True)


	@check_return('sess')
	def get_sess(self):
		return tf.Session(config=get_sess_config(), graph=self.g)


	def examples_to_samples(self, examples):
		"""
		Args:
			examples (list): [example1, ...]
		Returns:
			list: [(text_a, text_b, score), ...]
		"""
		return [(e.text_a, e.text_b, e.label) for e in examples]


	def save_and_cal_metric_for_examples(self, global_step, examples, processor, raw_to_true_texts, save_result_csv=False):
		mt = ModelTestor();reall_k_list = [1, 10]

		save_csv = self.config.predict_data_path.replace(os.path.join(DATA_PATH, 'preprocess', 'dataset'), self.RESULT_SAVE_FOLDER)
		prefix, postfix = os.path.splitext(save_csv)
		save_csv = prefix + f'-result-step{global_step}' + postfix
		os.makedirs(os.path.dirname(save_csv), exist_ok=True)

		if save_result_csv:
			processor.save_examples(save_csv, examples)
			metric_dict, raw_term_to_rank = mt.cal_metrics(save_csv, raw_to_true_texts, reall_k_list)
		else:
			samples = self.examples_to_samples(examples)
			metric_dict, raw_term_to_rank = mt.cal_metrics(samples, raw_to_true_texts, reall_k_list)
		tf.logging.info(metric_dict)
		json.dump(
			sorted([(rank, term) for term, rank in raw_term_to_rank.items()]),
			open(prefix + f'-term-rank-step{global_step}.json', 'w'),
			indent=2, ensure_ascii=False)
		json.dump(metric_dict, open(prefix + f'-metric-step{global_step}.json', 'w'), indent=2, ensure_ascii=False)
		for k, v in metric_dict.items():
			self.add_historys({k: v, f'{k}_STEP': global_step})
		simple_multi_line_plot(
			os.path.join(self.SAVE_FOLDER, 'pred_median_rank.png'),
			[self.history['MEDIAN_RANK_STEP']], [self.history['MEDIAN_RANK']],
			line_names=['Median Rank'], x_label='Step', y_label='Median Rank', title='Median Rank'
		)
		simple_multi_line_plot(
			os.path.join(self.SAVE_FOLDER, 'pred_recall_k.png'),
			[self.history[f'RECALL_{k}_STEP'] for k in reall_k_list], [self.history[f'RECALL_{k}'] for k in reall_k_list],
			line_names=[f'RECALL_{k}' for k in reall_k_list], x_label='Step', y_label='Recall_k', title='Recall at k'
		)
		self.save_history()


if __name__ == '__main__':
	pass
