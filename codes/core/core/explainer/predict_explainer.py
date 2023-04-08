

import json
from copy import deepcopy
from collections import Counter

from core.utils.utils import get_all_ancestors_with_dist, list_find, split_path
from core.utils.constant import PHELIST_REDUCE
from core.explainer.explainer import Explainer
from core.explainer.utils import add_tab, get_match_impre_noise_with_dist, obj2str, obj_to_str_with_max_depth


class SingleResultExplainer(Explainer):
	def __init__(self, model, pa_hpo_list, result, diag_list=None, top_n=None):
		super(SingleResultExplainer, self).__init__()
		self.model = model
		self.pa_hpo_list = pa_hpo_list
		self.result = result
		self.diag_list = [] if diag_list is None else diag_list
		self.top_n = len(result) if top_n is None else top_n
		self.tab = '\t'


	def explain(self):
		"""
		Returns:
			dict: {
				'PATIENT_HPO': [hpo_code, ...],
				'DIAGNOSIS_DISEASE': [{'DISEASE_CODE': dis_code, 'SCORE': score, 'RANK': rank}, ...],
				'PREDICT_DISEASE': [{'DISEASE_CODE': dis_code, 'SCORE': score}, ...],
				'DIAGNOSIS_DISEASE_DETAIL': [dis_item, ...]
				'PREDICT_DISEASE_DETAIL': [dis_item, ...]
			}, dis_item = {
				'RANK': int,
				'SCORE': float,
				'DISEASE': dis_code,
				'DISEASE_HPO': [{'CODE': hpo_code, 'PROBABILITY': probability}, ...]
				'DISEASE_GENE': [gene_code, ...],
				'MODEL_EXPLAIN': obj_list,
				'MATCH_IMPRE_SPE_OTH_RATE': [float, float, float, float],
				'PATIENT_MATCH': [{'CODE': hpo_code, 'PROBABILITY': probability}, ...],
				'PATIENT_IMPRE': {paHPO: {'DESCENDENT': disHPO, 'DISTANT': int, 'PROBABILITY': prob}), },
				'PATIENT_NOISE_SPE': {paHPO: {'ANCESTOR': disHPO, 'DISTANT': int, 'PROBABILITY': prob}, ...},
				'PATIENT_NOISE_OTH': [hpo_code, ],
			}
		"""
		ret_dict = {
			'PATIENT_HPO': self.pa_hpo_list,
			'DIAGNOSIS_DISEASE': [],
			'PREDICT_DISEASE': [{'DISEASE_CODE': item[0], 'SCORE': item[1]} for i, item in enumerate(self.result[:self.top_n]) ],
			'DIAGNOSIS_DISEASE_DETAIL': [],
			'PREDICT_DISEASE_DETAIL': [self.get_dis_item(rank) for rank in range(self.top_n)]
		}
		for diag_dis in self.diag_list:
			rank = list_find(self.result, lambda item: item[0] == diag_dis)
			assert rank >= 0
			item = self.result[rank]
			ret_dict['DIAGNOSIS_DISEASE'].append({'DISEASE_CODE': item[0], 'SCORE': item[1], 'RANK': rank})
			ret_dict['DIAGNOSIS_DISEASE_DETAIL'].append(self.get_dis_item(rank))
		return ret_dict


	def get_dis_item(self, rank):
		dis_code, score = self.result[rank]
		hpo_prob_dict = self.dis_to_hpo_prob_dict[dis_code]
		dis_hpo_list = self.hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)[dis_code]
		ret_dict = {
			'DISEASE': dis_code,
			'RANK': rank,
			'SCORE': score,
			'DISEASE_HPO': [{'CODE': hpo_code, 'PROBABILITY': hpo_prob_dict[hpo_code]} for hpo_code in dis_hpo_list],
			'DISEASE_GENE': self.hpo_reader.get_dis_to_gene().get(dis_code, []),
		}
		if self.model is not None:
			ret_dict['MODEL_EXPLAIN'] = add_tab(self.model.explain_as_str(self.pa_hpo_list, dis_code), self.tab*2).strip()
		mat, imp, noi_spe, noi_oth = get_match_impre_noise_with_dist(
			set(dis_hpo_list), self.pa_hpo_list, self.hpo_reader.get_slice_hpo_dict()
		)
		pa_hpo_len = len(self.pa_hpo_list)
		ret_dict['MATCH_IMPRE_SPE_OTH_RATE'] = [len(mat) / pa_hpo_len, len(imp) / pa_hpo_len, len(noi_spe) / pa_hpo_len, len(noi_oth) / pa_hpo_len]
		ret_dict['PATIENT_MATCH'] = [{'CODE': hpo_code, 'PROBABILITY': hpo_prob_dict[hpo_code]} for hpo_code in mat]
		ret_dict['PATIENT_IMPRE'] = {hpo: {'DESCENDENT': impre_tuple[0], 'DISTANT': impre_tuple[1], 'PROBABILITY': hpo_prob_dict[impre_tuple[0]]} for hpo, impre_tuple in imp }
		ret_dict['PATIENT_NOISE_SPE'] = {hpo: {'ANCESTOR': speTuple[0], 'DISTANT': speTuple[1], 'PROBABILITY': hpo_prob_dict[speTuple[0]]} for hpo, speTuple in noi_spe }
		ret_dict['PATIENT_NOISE_OTH'] = noi_oth

		return ret_dict


	def write_as_str(self, explain_dict):
		explain_dict = self.add_cns_info(explain_dict)
		explain_dict['PREDICT_DISEASE'] = self.to_str_list(explain_dict['PREDICT_DISEASE'])
		explain_dict['DIAGNOSIS_DISEASE'] = self.to_str_list(explain_dict['DIAGNOSIS_DISEASE'])

		explain_key_order = ['PATIENT_HPO', 'DIAGNOSIS_DISEASE', 'PREDICT_DISEASE', 'DIAGNOSIS_DISEASE_DETAIL', 'PREDICT_DISEASE_DETAIL']
		key_for_dis_item = [False, False, False, True, True]
		dis_item_key_order = [
			'RANK', 'SCORE', 'DISEASE', 'DISEASE_HPO', 'DISEASE_GENE', 'MODEL_EXPLAIN', 'MATCH_IMPRE_SPE_OTH_RATE',
			'PATIENT_MATCH', 'PATIENT_IMPRE', 'PATIENT_NOISE_SPE', 'PATIENT_NOISE_OTH'
		]
		ret_str = ''
		for k, is_dis_item in zip(explain_key_order, key_for_dis_item):
			child_str = self.dis_item_list_to_str(explain_dict[k], dis_item_key_order, 1) if is_dis_item else obj2str(explain_dict[k], 1, self.tab)
			ret_str += '{}{}'.format(self.tab*0+str(k)+':\n', child_str)
		return ret_str


	def dis_item_to_str(self, dis_item, key_order_list, depth):
		dis_item['DISEASE_HPO'] = self.to_str_list(dis_item['DISEASE_HPO'])
		dis_item['PATIENT_MATCH'] = self.to_str_list(dis_item['PATIENT_MATCH'])
		dis_item['PATIENT_IMPRE'] = self.to_str_list(dis_item['PATIENT_IMPRE'])
		dis_item['PATIENT_NOISE_SPE'] = self.to_str_list(dis_item['PATIENT_NOISE_SPE'])
		return ''.join(['{}{}'.format(self.tab*depth+str(k)+':\n', obj2str(dis_item[k], depth+1, self.tab)) for k in key_order_list])


	def dis_item_list_to_str(self, dis_item_list, key_order_list, depth):
		return ''.join([self.dis_item_to_str(dis_item, key_order_list, depth) for dis_item in dis_item_list])


class MultiModelExplainer(Explainer):
	def __init__(self, pa_hpo_list, models, results, diag_lists=None, top_n_list=None):
		super(MultiModelExplainer, self).__init__()
		self.pa_hpo_list = pa_hpo_list
		self.models = models
		self.results = results
		self.diag_lists = diag_lists if diag_lists is not None else []
		self.top_n_list = top_n_list if top_n_list is not None else [1, 5, 10]
		self.tab = '\t'


	def set_pa_hpo_list(self, pa_hpo_list):
		self.pa_hpo_list = pa_hpo_list


	def add_result(self, model, result, diag=None):
		self.models.append(model)
		self.results.append(result)
		diag = [] if diag is None else diag
		self.diag_lists.append(diag)


	def explain(self):
		"""
		Args:
			top_n_list (list)
		Returns:
			dict: {
				'TOP_{}_VOTE': {
					'DISEASE': [(dis_code, count), ...]
					'GENE': [(gene_code, count), ...]
				}
			}
		"""
		return {'TOP_{}_VOTE'.format(top_n): self.explain_top_n(top_n) for top_n in self.top_n_list}


	def write_as_str(self, explain_dict):
		explain_dict = self.add_cns_info(explain_dict)
		for k, d in explain_dict.items():
			d['DISEASE'] = [str(obj) for obj in d['DISEASE']]
			d['GENE'] = [str(obj) for obj in d['GENE']]
		return obj_to_str_with_max_depth(explain_dict, 0, self.tab)


	def explain_top_n(self, top_n):
		return {
			'DISEASE': self.vote_disease(top_n),
			'GENE': self.vote_genes(top_n),
		}


	def vote_disease(self, top_n):
		"""
		Returns:
			list: [(dis_code, count), ...]
		"""
		counter = Counter()
		for result in self.results:
			counter.update([dis_code for dis_code, score in result[:top_n]])
		return sorted(counter.items(), key=lambda item: item[1], reverse=True)


	def vote_genes(self, top_n):
		"""
		Returns:
			list: [(geneSymbol, count), ...]; ordered by count, big followed by small
		"""
		self.hpo_reader.get_dis_to_gene()
		counter = Counter()
		for result in self.results:
			for dis_code, score in result[:top_n]:
				counter.update(self.hpo_reader.get_dis_to_gene().get(dis_code, []))
		return sorted(counter.items(), key=lambda item: item[1], reverse=True)


class MultiResultExplainer(Explainer):
	def __init__(self):
		super(MultiResultExplainer, self).__init__()
		
		

class CompareModelExplainer(Explainer):
	def __init__(self):
		super(CompareModelExplainer, self).__init__()



if __name__ == '__main__':
	pass


