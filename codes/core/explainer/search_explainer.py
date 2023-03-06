

import json
import numpy as np
import os
from tqdm import tqdm

from core.utils.utils import unique_list, check_return
from core.explainer.explainer import Explainer
from core.explainer.utils import obj2str


class SearchExplainer(Explainer):
	def __init__(self, texts, search_results):
		"""
		Args:
			texts (list): [str, ...]
			search_results (list): [(hpo_list, pos_list), ...]; hpo_list=[hpo_code1, hpo_code2, ...]; pos_list=[np.array([begin1, end1]), ...]
		"""
		super(SearchExplainer, self).__init__()
		self.texts = texts
		self.search_results = search_results
		self.hpo2text = None


	def explain(self):
		"""
		Returns:
			dict: {
				'AVERAGE_CODE_NUM': int,
				'MAX_CODE_NUM': int,
				'MIN_CODE_NUM': int,
				'AVERAGE_UNIQUE_CODE_NUM': int,
				'MIN_UNIQUE_CODE_NUM': int,
				'MAX_UNIQUE_CODE_NUM': int,
				'TOTAL_CODE_NUM': int,
				'TOTAL_UNIQUE_CODE_NUM': int
			}
		"""
		d = {}
		code_nums = [len(hpo_list) for hpo_list, _ in self.search_results]
		d['MIN_CODE_NUM'], d['MAX_CODE_NUM'], d['AVERAGE_CODE_NUM'], d['TOTAL_CODE_NUM'] = min(code_nums), max(code_nums), np.mean(code_nums), sum(code_nums)
		code_nums = [len(unique_list(hpo_list)) for hpo_list, _ in self.search_results]
		d['MIN_UNIQUE_CODE_NUM'], d['MAX_UNIQUE_CODE_NUM'], d['AVERAGE_UNIQUE_CODE_NUM'] = min(code_nums), max(code_nums), np.mean(code_nums)
		d['TOTAL_UNIQUE_CODE_NUM'] = len(unique_list([hpo for hpo_list, _ in self.search_results for hpo in hpo_list]))
		return d


	def write_as_str(self, explain_dict):
		def get_single(text, search_result):
			ret_str = ''
			match_hpos, pos_list = search_result
			ret_str += '----------------------------------------------\n{}\n'.format(text)
			for hpo, pos in zip(match_hpos, pos_list):
				ret_str += '{}, {} -> {}({})\n'.format(pos, text[pos[0]: pos[1]], hpo, chpo_dict.get(hpo, {}).get('CNS_NAME', ''))
			return ret_str + '\n'
		ret_str = obj2str(explain_dict)
		chpo_dict = self.hpo_reader.get_chpo_dict()
		ret_str += ''.join([get_single(text, search_result) for text, search_result in zip(self.texts, self.search_results)])
		return ret_str


	def write_as_anns(self, ann_paths):
		"""
		Args:
			ann_paths (list): [ann_path, ...]
		"""
		for text, search_result, ann_path in tqdm(zip(self.texts, self.search_results, ann_paths)):
			ann_list = self.search_result_to_ann_list(text, search_result)
			self.ann_list_to_ann_file(text, ann_list, ann_path)


	@check_return('hpo2text')
	def get_hpo2text(self, hpo_reader):
		hpo_dict = hpo_reader.get_hpo_dict()
		chpo_dict = hpo_reader.get_chpo_dict()
		ret_dict = {}
		for hpo in hpo_dict:
			ret_dict[hpo] = chpo_dict[hpo]['CNS_NAME'] if hpo in chpo_dict else hpo_dict[hpo]['ENG_NAME']
		return ret_dict


	def search_result_to_ann_list(self, text, search_result):
		"""
		Args:
			text (str)
			search_result (tuple): (hpo_list, pos_list) = ([hpo_code1, hpo_code2, ...], [np.array([begin1, end1]), ...])
		Returns:
			list: [
				{
					'SPAN_LIST': [(start_pos, end_pos), ...],
					'SPAN_TEXT': str,
					'HPO_CODE': str,
					'HPO_TEXT': str,
					'TAG_TYPE': str
				},
				...
			]
		"""
		hpo2text = self.get_hpo2text(self.hpo_reader)
		hpo_list, span_ary_list = search_result
		assert len(hpo_list) == len(span_ary_list)
		ret_list = []
		for hpo, span_ary in zip(hpo_list, span_ary_list):
			if not hpo:
				print(text)
				print(search_result)
				print(text[span_ary[0]: span_ary[1]])
			ret_list.append({
				'SPAN_LIST': [span_ary.tolist(), ],
				'SPAN_TEXT': text[span_ary[0]: span_ary[1]],
				'HPO_CODE': hpo,
				'HPO_TEXT': hpo2text[hpo],
				'TAG_TYPE': 'Phenotype'
			})
		return ret_list


	def ann_list_to_ann_file(self, text, ann_list, ann_path):
		"""
		Args:
			text (str)
			ann_list (list): return of search_result_to_ann_list
			ann_path (str)
		"""
		t_idx, n_idx = 0, 0
		lines = []
		for d in ann_list:
			if len(d['HPO_CODE'].strip()) == 0: continue
			span_str = ';'.join(f'{start} {end}' for start, end in d['SPAN_LIST'])
			lines.append('T{}\t{}\t{}\n'.format(
				t_idx,
				'{} {}'.format(d['TAG_TYPE'], span_str),
				' '.join([text[start: end] for start, end in d['SPAN_LIST']])
			))
			lines.append('N{}\t{}\t{}\n'.format(
				n_idx,
				'Reference T{} brat_hpo:{}'.format(t_idx, d['HPO_CODE']),
				d['HPO_TEXT']
			))
			t_idx += 1; n_idx += 1
		os.makedirs(os.path.dirname(ann_path), exist_ok=True)
		open(ann_path, 'w').write(''.join(lines))


if __name__ == '__main__':
	pass





