

import os
import json
import tensorflow as tf
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from bert_syn.bert_pkg import tokenization
from bert_syn.utils.utils import timer


# ============================================================================
class InputExample(object):
	"""A single training/test example for simple sequence classification."""
	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


	def __str__(self):
		return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


# ============================================================================
class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


# ============================================================================
class DataProcessor(object):
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer


	def get_examples(self, path, guid_mark):
		"""
		Args:
			path (string): endswith '.csv'
			mark (str): prefix of guid
		Returns:
			list: [InputExample, ...]
		"""
		raise NotImplementedError()


	def get_labels(self):
		return ["0", "1"]


	def get_tfrecord_path(self, src_path, max_seq_len):
		if os.path.isdir(src_path):
			return f'{src_path}-{max_seq_len}.tfrecord'
		return '{}-{}.tfrecord'.format(os.path.splitext(src_path)[0], max_seq_len)


	def get_info_json(self, tfrecord_path):
		return os.path.splitext(tfrecord_path)[0] + '-tfrecord-info.json'


	def get_data_info(self, src_path, max_seq_len):
		return json.load(open(self.get_info_json(self.get_tfrecord_path(src_path, max_seq_len))))


	def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
		"""Truncates a sequence pair in place to the maximum length."""

		# This is a simple heuristic which will always truncate the longer sequence
		# one token at a time. This makes more sense than truncating an equal percent
		# of tokens from each, since if one sequence is very short then each token
		# that's truncated likely contains more information than a longer sequence.
		while True:
			total_length = len(tokens_a) + len(tokens_b)
			if total_length <= max_length:
				break
			if len(tokens_a) > len(tokens_b):
				tokens_a.pop()
			else:
				tokens_b.pop()


	def convert_single_example_wrapper(self, paras):
		return self.convert_single_example(*paras)


	def convert_single_example_string_wrapper(self, paras):
		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f
		feature = self.convert_single_example(*paras)
		features = collections.OrderedDict()
		features["input_ids"] = create_int_feature(feature.input_ids)
		features["input_mask"] = create_int_feature(feature.input_mask)
		features["segment_ids"] = create_int_feature(feature.segment_ids)
		features["label_ids"] = create_int_feature([feature.label_id])

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		return tf_example.SerializeToString()


	def convert_single_example(self, ex_index, example, label_list, max_seq_length, tokenizer, verbose=False):
		"""Converts a single `InputExample` into a single `InputFeatures`."""
		label_map = {}
		for (i, label) in enumerate(label_list):
			label_map[label] = i

		tokens_a = tokenizer.tokenize(example.text_a)
		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if tokens_b:
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0     0   0   0  0     0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		label_id = label_map[example.label]
		if verbose and ex_index < 5:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens: %s" % " ".join(
				[tokenization.printable_text(x) for x in tokens]))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

		feature = InputFeatures(
			input_ids=input_ids,
			input_mask=input_mask,
			segment_ids=segment_ids,
			label_id=label_id,)
		return feature


	@timer
	def convert_examples_to_feed_dict(self, examples, label_list, max_seq_length, tokenizer, cpu_use=1, chunk_size=2000):
		"""
		Returns:
			dict: {
				'input_ids': [input_ids1, input_ids2, ...],
				'input_mask': [input_mask1, input_mask2],
				'segment_ids': [segment_ids1, segment_ids2, ...]
				'label_ids': [label_id1, label_id2, ...]
			}
		"""
		def get_iterator(examples):
			for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
				yield ex_index, example, label_list, max_seq_length, tokenizer

		def add_feature(feed_dict, feature):
			feed_dict["input_ids"].append(feature.input_ids)
			feed_dict["input_mask"].append(feature.input_mask)
			feed_dict["segment_ids"].append(feature.segment_ids)
			feed_dict["label_ids"].append(feature.label_id)

		feed_dict = {"input_ids": [], "input_mask": [], "segment_ids": [], "label_ids": []}
		paras = get_iterator(examples)
		if cpu_use == 1:
			for para in paras:
				add_feature(feed_dict, self.convert_single_example_wrapper(para))
		else:
			with Pool(cpu_use) as pool:
				for feature in pool.imap(self.convert_single_example_wrapper, paras, chunksize=chunk_size):
					add_feature(feed_dict, feature)
		for k in feed_dict:
			feed_dict[k] = np.array(feed_dict[k], np.int32)
		return feed_dict


	def file_based_convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer,
			output_file, cpu_use=12, chunk_size=2000):
		"""Convert a set of `InputExample`s to a TFRecord file."""
		def get_iterator(examples):
			for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
				yield ex_index, example, label_list, max_seq_length, tokenizer

		writer = tf.python_io.TFRecordWriter(output_file)
		paras = get_iterator(examples)
		if cpu_use == 1:
			for para in paras:
				writer.write(self.convert_single_example_string_wrapper(para))
		else:
			with Pool(cpu_use) as pool:
				for tf_example_string in pool.imap(self.convert_single_example_string_wrapper, paras, chunksize=chunk_size):
					writer.write(tf_example_string)
		writer.close()


	def tfrecord_to_dataset(self, tfrecord_path, seq_length, is_training, drop_remainder, batch_size, buffer_size=100):
		"""Creates an `input_fn` closure to be passed to TPUEstimator."""
		name_to_features = {
			"input_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"input_mask":tf.FixedLenFeature([seq_length], tf.int64),
			"segment_ids":tf.FixedLenFeature([seq_length], tf.int64),
			"label_ids":tf.FixedLenFeature([], tf.int64),
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
				drop_remainder=drop_remainder))

		return d


	@timer
	def get_tfrecord(self, src_path, guid_mark, max_seq_len, cpu_use=12, chunk_size=200):
		"""
		Returns:
			str: tfrecord_path
			dict: information
		"""
		tfrecord_path = self.get_tfrecord_path(src_path, max_seq_len)
		info_json = self.get_info_json(tfrecord_path)
		if not os.path.exists(tfrecord_path):
			examples = self.get_examples(src_path, guid_mark)
			self.file_based_convert_examples_to_features(examples, self.get_labels(), max_seq_len, self.tokenizer,
				tfrecord_path, cpu_use=cpu_use, chunk_size=chunk_size)
			info = {'n_samples':len(examples), 'max_seq_len':max_seq_len}
			json.dump(info, open(info_json, 'w'), indent=2)
			return tfrecord_path, info
		tf.logging.info('tfrecord file existed, load directly: {}'.format(tfrecord_path))
		return tfrecord_path, json.load(open(info_json))


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
			label = str(line_data[2])
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


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
			label = str(sample[2]) if len(sample) >= 3 else '0'
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


if __name__ == '__main__':
	from bert_syn.utils.constant import DATA_PATH
	processor = CsvExampleGenerator()
	examples = processor.get_examples(os.path.join(DATA_PATH, 'chpo', 'train.csv'), 'train')
	print(len(examples), examples[0], examples[-1])

	example = InputExample(guid='train-1', text_a='aaa', text_b='bbb', label='0')
	print(example)
