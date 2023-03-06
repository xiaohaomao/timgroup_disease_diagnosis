

import os
import json
import tensorflow as tf
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from copy import deepcopy

from bert_syn.bert_pkg import tokenization
from bert_syn.core.input import InputExample
from bert_syn.core.input import DataProcessor as DataProcessorBase
from bert_syn.utils.utils import timer, equal_to


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, text_a_input_ids, text_a_input_mask, text_a_segment_ids,
			text_b_input_ids, text_b_input_mask, text_b_segment_ids, label):
		self.text_a_input_ids = text_a_input_ids
		self.text_a_input_mask = text_a_input_mask
		self.text_a_segment_ids = text_a_segment_ids
		self.text_b_input_ids = text_b_input_ids
		self.text_b_input_mask = text_b_input_mask
		self.text_b_segment_ids = text_b_segment_ids
		self.label = label


class DataProcessor(DataProcessorBase):
	def __init__(self, tokenizer):
		super(DataProcessor, self).__init__(tokenizer)


	def get_labels(self):
		return [-1., 1., 0.0]


	def get_tfrecord_path(self, src_path, max_seq_len):
		if os.path.isdir(src_path):
			return f'{src_path}-{"ddml"}-{max_seq_len}.tfrecord'
		return '{}-{}-{}.tfrecord'.format(os.path.splitext(src_path)[0], 'ddml', max_seq_len)


	def convert_single_example(self, ex_index, example, label_list, max_seq_length, tokenizer, verbose=False):
		text_a_example = deepcopy(example); text_a_example.text_b = None
		text_a_feature = super(DataProcessor, self).convert_single_example(ex_index, text_a_example, label_list, max_seq_length, tokenizer, verbose=verbose)

		text_b_example = deepcopy(example); text_b_example.text_a = text_b_example.text_b; text_b_example.text_b = None
		text_b_feature = super(DataProcessor, self).convert_single_example(ex_index, text_b_example, label_list, max_seq_length, tokenizer, verbose=verbose)

		return InputFeatures(
			text_a_input_ids=text_a_feature.input_ids,
			text_a_input_mask=text_a_feature.input_mask,
			text_a_segment_ids=text_a_feature.segment_ids,
			text_b_input_ids=text_b_feature.input_ids,
			text_b_input_mask=text_b_feature.input_mask,
			text_b_segment_ids=text_b_feature.segment_ids,
			label=float(example.label)
		)


	def convert_examples_to_feed_dict(self, examples, label_list, max_seq_length, tokenizer, cpu_use=1, chunk_size=2000, ignore_text_b=False):
		"""
		Returns:
			dict: {
				'input_ids': [input_ids1, input_ids2, ...],
				'input_mask': [input_mask1, input_mask2],
				'segment_ids': [segment_ids1, segment_ids2, ...]
				'label': [label1, label2, ...]
			}
		"""
		def get_iterator(examples):
			for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
				yield ex_index, example, label_list, max_seq_length, tokenizer

		def add_feature(feed_dict_a, feed_dict_b, feature):
			feed_dict_a["input_ids"].append(feature.text_a_input_ids)
			feed_dict_a["input_mask"].append(feature.text_a_input_mask)
			feed_dict_a["segment_ids"].append(feature.text_a_segment_ids)
			feed_dict_a["label"].append(feature.label)
			if not ignore_text_b:
				feed_dict_b["input_ids"].append(feature.text_b_input_ids)
				feed_dict_b["input_mask"].append(feature.text_b_input_mask)
				feed_dict_b["segment_ids"].append(feature.text_b_segment_ids)

		feed_dict_a = {"input_ids": [], "input_mask": [], "segment_ids": [], "label": []}
		feed_dict_b = deepcopy(feed_dict_a)
		paras = get_iterator(examples)
		if cpu_use == 1:
			for para in paras:
				add_feature(feed_dict_a, feed_dict_b, self.convert_single_example_wrapper(para))
		else:
			with Pool(cpu_use) as pool:
				for feature in pool.imap(self.convert_single_example_wrapper, paras, chunksize=chunk_size):
					add_feature(feed_dict_a, feed_dict_b, feature)
		for k in ('input_ids', 'input_mask', 'segment_ids'):
			feed_dict_a[k].extend(feed_dict_b[k])
			feed_dict_a[k] = np.array(feed_dict_a[k], np.int32)
		feed_dict_a['label'] = np.array(feed_dict_a['label'], np.float32)
		return feed_dict_a


	def convert_single_example_string_wrapper(self, paras):
		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f
		def create_float_feature(values):
			return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

		feature = self.convert_single_example(*paras)
		features = collections.OrderedDict()
		features["text_a_input_ids"] = create_int_feature(feature.text_a_input_ids)
		features["text_a_input_mask"] = create_int_feature(feature.text_a_input_mask)
		features["text_a_segment_ids"] = create_int_feature(feature.text_a_segment_ids)
		features["text_b_input_ids"] = create_int_feature(feature.text_b_input_ids)
		features["text_b_input_mask"] = create_int_feature(feature.text_b_input_mask)
		features["text_b_segment_ids"] = create_int_feature(feature.text_b_segment_ids)
		features["label"] = create_float_feature([feature.label])

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		return tf_example.SerializeToString()


	def tfrecord_to_dataset(self, tfrecord_path, seq_length, is_training, drop_remainder, batch_size, buffer_size=100, drop_text_b=False):
		"""Creates an `input_fn` closure to be passed to TPUEstimator."""
		name_to_features = {
			"text_a_input_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"text_a_input_mask":tf.FixedLenFeature([seq_length], tf.int64),
			"text_a_segment_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"text_b_input_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"text_b_input_mask":tf.FixedLenFeature([seq_length], tf.int64),
			"text_b_segment_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"label":tf.FixedLenFeature([], tf.float32),
		}

		def _decode_record(record, name_to_features):
			"""Decodes a record to a TensorFlow example."""
			example = tf.parse_single_example(record, name_to_features)

			# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
			# So cast all int64 to int32.
			for name in list(example.keys()):
				t = example[name]
				if t.dtype == tf.int64:
					t = tf.to_int32(t)
				example[name] = t

			return example

		def reorder_batch(data):
			text_a_input_ids = data["text_a_input_ids"]
			text_a_input_mask = data["text_a_input_mask"]
			text_a_segment_ids = data["text_a_segment_ids"]
			text_b_input_ids = data["text_b_input_ids"]
			text_b_input_mask = data["text_b_input_mask"]
			text_b_segment_ids = data["text_b_segment_ids"]
			label = data["label"]

			return {
				"input_ids": text_a_input_ids if drop_text_b else tf.concat([text_a_input_ids, text_b_input_ids], axis=0),
				"input_mask": text_a_input_mask if drop_text_b else tf.concat([text_a_input_mask, text_b_input_mask], axis=0),
				"segment_ids": text_a_segment_ids if drop_text_b else tf.concat([text_a_segment_ids, text_b_segment_ids], axis=0),
				"label": label
			}

		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		d = tf.data.TFRecordDataset(tfrecord_path)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=buffer_size)

		d = d.apply(
			tf.contrib.data.map_and_batch(
				lambda record:_decode_record(record, name_to_features),
				batch_size=batch_size,
				drop_remainder=drop_remainder)
		).map(reorder_batch)

		return d


# ============================================================================
class CsvDataProcessor(DataProcessor):
	def __init__(self, tokenizer):
		super(CsvDataProcessor, self).__init__(tokenizer)


	def get_examples(self, path, guid_mark):
		df = pd.read_csv(path, encoding='utf-8')
		examples = []
		for i, line_data in enumerate(df.values):
			guid = f'{guid_mark}-{i}'
			text_a = tokenization.convert_to_unicode(str(line_data[0]))
			text_b = tokenization.convert_to_unicode(str(line_data[1]))
			label = float(line_data[2])
			if equal_to(label, 0.0):
				label = -1.
			if equal_to(label, 0.5):
				label = 0.
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


	@timer
	def save_examples(self, path, examples):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		pd.DataFrame(
			[{'text_a': e.text_a, 'text_b': e.text_b, 'label': e.label} for e in examples],
			columns=['text_a', 'text_b', 'label']).to_csv(path, index=False)


# ============================================================================
class MemDataProcessor(DataProcessor):
	def __init__(self, tokenizer):
		super(MemDataProcessor, self).__init__(tokenizer)


	def get_examples(self, samples, guid_mark):
		"""
		Args:
			samples (list): [(sent1, sent2, label), ...] or [(sent1, sent2), ...]
			guid_mark:
		Returns:
		"""
		examples = []
		for i, sample in enumerate(samples):
			guid = f'{guid_mark}-{i}'
			text_a = tokenization.convert_to_unicode(str(sample[0]))
			text_b = tokenization.convert_to_unicode(str(sample[1]))
			label = sample[2] if len(sample) >= 3 else -1.
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


if __name__ == '__main__':
	pass