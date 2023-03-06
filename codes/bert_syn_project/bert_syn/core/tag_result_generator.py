

import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from ACtrie import ACtrie
from bert_syn.core.data_helper import PUMCReader
from bert_syn.core.data_helper import HPOReader
from bert_syn.utils.constant import RESULT_PATH, TEST_DATA, VALIDATION_DATA, DATA_PATH
from bert_syn.utils.utils import dict_list_add, timer


class TagResultGenrator(object):
	def __init__(self, name=None, save_folder=None):
		self.name = name or 'AlbertDDML'
		self.SAVE_FOLDER = save_folder or os.path.join(DATA_PATH, 'preprocess', 'pumc_87', f'doc-{self.name}')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)


	@timer
	def gen_tag_jsons(self, word2hpos, data_type):
		"""
		Args:
			word2hpos (dict): {word: [hpo1, hpo2, ...]}
			data_type: TEST_DATA | VALIDATION_DATA
			score_thres (float or None): only keep matching result with score > score_thres
		"""
		pumc_jsons = PUMCReader().get_json_paths(data_type)
		for json_path in pumc_jsons:
			self.gen_single_tag_json(
				word2hpos, json.load(open(json_path)),
				os.path.join(self.SAVE_FOLDER, os.path.split(json_path)[1]))


	@timer
	def get_word2hpos_from_match_result(self, result, score_thres=None):
		"""
		Args:
			result (str or list):
				str: csv_path, columns=('text_a', 'text_b', 'score')
				list: [(text_a, text_b, score), ...]
			score_thres (float or None): only keep matching result with score > score_thres
		Returns:
			dict: {raw_term: [hpo_term1, hpo_term2]}
		"""
		score_thres = score_thres or -np.inf
		if isinstance(result, str):
			result = pd.read_csv(result).values
		tgt_term_to_codes = HPOReader().get_cns_to_hpo()
		return {src_term: tgt_term_to_codes[tgt_term] for src_term, tgt_term, score in result if score > score_thres}


	def get_word2hpos_from_model(self, terms, model_name, global_step=None, score_thres=None):
		"""
		Args:
			terms (str or list):
		Returns:
			dict: {term: [hpo1, hpo2, ...]}
		"""
		def process_terms(terms):
			if isinstance(terms, str):
				if terms.endswith('.json'):
					terms = json.load(open(terms))
				else:
					assert terms.endswith('.txt')
					terms = open(terms).read().strip().splitlines()
			return [t.strip() for t in terms if t.strip()]
		from bert_syn.core.bert_ddml_sim import BertDDMLSim
		bert_sim = BertDDMLSim(model_name)
		bert_sim.restore(global_step=global_step)

		hpo_reader = HPOReader()
		tgt_terms = hpo_reader.get_cns_list()
		tgt_term_to_codes = hpo_reader.get_cns_to_hpo()
		bert_sim.set_dict_terms(tgt_terms)

		score_thres = score_thres or -np.inf
		src_terms = process_terms(terms)
		tgt_term_score_pairs = bert_sim.predict_best_match(terms)
		return {src_term: tgt_term_to_codes[tgt_term] for src_term, (tgt_term, score) in zip(src_terms, tgt_term_score_pairs) if score > score_thres}


	def gen_single_tag_json(self, word2hpos, field_to_info, save_json):
		"""
		Args:
			word2hpos (dict): {term: [hpo1, hpo2]}
			field_to_info (dict or str): {
				FIELD: {
					'RAW_TEXT': str,
					'ENTITY_LIST': [
						{
							'SPAN_LIST': [(start_pos, end_pos), ...],
							'SPAN_TEXT': str
							'HPO_CODE': str,
							'HPO_TEXT': str,
							'TAG_TYPE': str
						},
						...
					]
				}
			}
			save_json (str)
			ac (ACtire)
		"""
		if isinstance(field_to_info, str):
			assert field_to_info.endswith('.json')
			field_to_info = json.load(open(field_to_info))
		assert isinstance(field_to_info, dict)
		hpo_to_term = HPOReader().get_hpo_to_cns()
		for field_name, field_info in field_to_info.items():
			new_entity_list = []
			for entity_item in field_info['ENTITY_LIST']:
				src_term = entity_item['SPAN_TEXT'].replace(' ', '')
				for hpo in word2hpos.get(src_term, []):
					new_entity_list.append({
						'SPAN_LIST': entity_item['SPAN_LIST'],
						'SPAN_TEXT': entity_item['SPAN_TEXT'],
						'HPO_CODE': hpo,
						'HPO_TEXT': hpo_to_term[hpo]
					})
			field_info['ENTITY_LIST'] = new_entity_list
		os.makedirs(os.path.dirname(save_json), exist_ok=True)
		json.dump(field_to_info, open(save_json, 'w'), indent=2, ensure_ascii=False)


	def init_ac(self, terms):
		ac = ACtrie()
		for term in terms:
			ac.insert(term)
		return ac


	def tag_text(self, text, ac):
		"""
		Args:
			text (str)
			ac (ACtire)
		Returns:
			list: [(start_pos, match_term)]
		"""
		offset_term_list = ac.match_all(text)
		return list(set(offset_term_list))



if __name__ == '__main__':
	from bert_syn.core.baseline import DictSim
	from bert_syn.core.data_helper import SynDictReader


	# ===================================================================================================
	syn_dict_name = 'chpo'
	model_name = 'jaccard'
	data_types = [TEST_DATA, VALIDATION_DATA]

	terms = json.load(open(os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'test', 'ehr_terms_all.json')))
	terms = terms + json.load(open(os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'eval', 'ehr_terms_all.json')))
	terms = list(set(terms))

	if syn_dict_name == 'chpo':
		hpo_to_cns = HPOReader().get_hpo_to_cns()
		hpo_to_syn_terms = {hpo: [term] for hpo, term in hpo_to_cns.items()}
	elif syn_dict_name == 'chpo_umls_source':
		hpo_to_syn_terms = SynDictReader().get_hpo_to_source_syn_terms()
	elif syn_dict_name == 'chpo_umls_bg_source':
		hpo_to_syn_terms = SynDictReader().get_hpo_to_syn_terms_with_bg_evaluate()
	else:
		raise RuntimeError('Unknown syn dict name: {}'.format(syn_dict_name))

	dict_sim = DictSim('jaccard', hpo_to_syn_terms, match_type='jaccard')
	hpo_term_score_pairs = dict_sim.predict_best(terms)
	samples = [(term, hpo_term, score) for term, (hpo_term, score) in zip(terms, hpo_term_score_pairs)]

	trg = TagResultGenrator(model_name)
	word2hpos = trg.get_word2hpos_from_match_result(samples)
	for data_type in data_types:
		trg.gen_tag_jsons(word2hpos, data_type)

		

