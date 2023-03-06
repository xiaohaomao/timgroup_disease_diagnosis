
import os
import json
from collections import Counter
import numpy as np
import pandas as pd

from bert_syn.utils.utils import cal_quartile
from bert_syn.utils.utils_draw import simple_dist_plot

class Explainer(object):
	def __init__(self):
		super(Explainer, self).__init__()


	def explain(self):
		raise NotImplementedError()


class SampleExplainer(object):
	def __init__(self, samples=None, csv_path=None):
		"""
		Args:
			samples (list): [(text_a, text_b, label), ...]
		"""
		if samples is None:
			self.samples = pd.read_csv(csv_path).values.tolist()
		else:
			self.samples = samples


	def explain(self):
		samples = self.samples
		info_dict = {'SAMPLE_NUM':len(samples)}
		label_counter = Counter([label for text_a, text_b, label in samples])
		info_dict['LABEL_COUNT'] = label_counter.most_common()
		len_list = [len(text) for sample in self.samples for text in sample[:2]]
		info_dict['TEXT_LEN_QUATILE'] = cal_quartile(len_list)
		info_dict['TEXT_LEN_COUNT'] = Counter(len_list).most_common()
		pair_len_list = [len(sample[0])+len(sample[1]) for sample in self.samples]
		info_dict['PAIR_LEN_QUATILE'] = cal_quartile(pair_len_list)
		info_dict['PAIR_LEN_COUNT'] = Counter(pair_len_list).most_common()
		return info_dict


	def explain_save(self, json_path):
		os.makedirs(os.path.dirname(json_path), exist_ok=True)
		json.dump(self.explain(), open(json_path, 'w'), indent=2)
		self.draw_str_len_dist(
			os.path.splitext(json_path)[0] + '-text-len-dist.png',
			[len(text) for sample in self.samples for text in sample[:2]])
		self.draw_str_len_dist(
			os.path.splitext(json_path)[0] + '-pair-len-dist.png',
			[len(sample[0])+len(sample[1]) for sample in self.samples])


	def draw_str_len_dist(self, figpath, len_list):
		simple_dist_plot(figpath, len_list, bins=20, x_label='Text length', title='Text length dist')


if __name__ == '__main__':
	pass