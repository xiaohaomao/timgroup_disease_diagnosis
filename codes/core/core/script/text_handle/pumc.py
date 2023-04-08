import os
import json

from core.reader import HPOReader, HPOFilterDatasetReader
from core.utils.utils import get_file_list, get_all_descendents_for_many
from core.utils.constant import DATA_PATH
from core.text_handler.max_inv_text_searcher import MaxInvTextSearcher
from core.text_handler.term_matcher import ExactTermMatcher, BagTermMatcher
from core.text_handler.syn_generator import UMLSSynGenerator

def get_hpo_2_text(hpo_reader):
	hpo_dict = hpo_reader.get_hpo_dict()
	chpo_dict = hpo_reader.get_chpo_dict()
	ret_dict = {}
	for hpo in hpo_dict:
		ret_dict[hpo] = chpo_dict[hpo]['CNS_NAME'] if hpo in chpo_dict else hpo_dict[hpo]['ENG_NAME']
	return ret_dict


def get_remove_hpo_set(hpo_dict):
	return get_all_descendents_for_many(['HP:0000005', 'HP:0012823', 'HP:0040006', 'HP:0040279'], hpo_dict)

def remove_hpos(hpo2terms, rm_hpo_set):
	return {hpo: terms for hpo, terms in hpo2terms.items() if hpo not in rm_hpo_set}

def remove_terms(hpo2terms, rm_term_set):
	return {hpo: [term for term in terms if term not in rm_term_set] for hpo, terms in hpo2terms.items()}


def combine_hpo_2_terms(hpo_to_terms1, hpo_to_terms2):
	hpo_set = set(hpo_to_terms1.keys()) | set(hpo_to_terms2.keys())
	return {hpo: list(set(hpo_to_terms1.get(hpo, []) + hpo_to_terms2.get(hpo, []))) for hpo in hpo_set}


def gen_entity_list(text, searcher, hpo2text):
	"""
	Args:
		searcher:
		text (str)
		hpo2text (dict): {hpo_code: str}
	Returns:
		list: [
			{
				'SPAN_LIST': [(start_pos, end_pos), ...],
				'SPAN_TEXT': str
				'HPO_CODE': str,
				'HPO_TEXT': str,
				'TAG_TYPE': str
			}
		]
	"""
	hpo_list, span_ary_list = searcher.search(text)
	assert len(hpo_list) == len(span_ary_list)
	ret_list = []
	for hpo, span_ary in zip(hpo_list, span_ary_list):
		ret_list.append({
			'SPAN_LIST':[span_ary.tolist(),],
			'SPAN_TEXT': text[span_ary[0]: span_ary[1]],
			'HPO_CODE': hpo,
			'HPO_TEXT': hpo2text[hpo],
			'TAG_TYPE': 'Phenotype'
		})
	return ret_list


def handle_single(field2text, searcher, hpo2text):
	"""
	Args:
		field2text (dict): {field: text}
		hpo2text (dict)
	Returns:
		dict: {
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
	"""
	ret_dict = {}
	for field, text in field2text.items():
		ret_dict[field] = {
			'RAW_TEXT': text,
			'ENTITY_LIST': gen_entity_list(text, searcher, hpo2text)
		}
	return ret_dict


def process(input_folder, output_folder):
	hpo_reader = HPOReader()
	hpo2text = get_hpo_2_text(hpo_reader)
	hpo_set_remove = get_remove_hpo_set(hpo_reader.get_hpo_dict())
	terms_remove = {'发作', '甲状腺', '方相', '脱落', '发红', '发白', '直肠', '前间隙', '盆腔', '积水', '开口'}
	sg = UMLSSynGenerator()

	all_syn_dict = {
		'CHPO': remove_terms(remove_hpos({hpo: [info_dict['CNS_NAME']] for hpo, info_dict in hpo_reader.get_chpo_dict().items()}, hpo_set_remove), terms_remove),
		'CHPO_MANUAL': remove_terms(remove_hpos(sg.get_hpo_to_source_syn_terms(), hpo_set_remove), terms_remove),
		'CHPO_MANUAL_BG': remove_terms(remove_hpos(combine_hpo_2_terms(sg.get_hpo_to_source_syn_terms(), sg.get_hpo_to_syn_terms_with_bg_evaluate()), hpo_set_remove), terms_remove),
	}
	matchers = []
	for syn_dict_name, syn_dict in all_syn_dict.items():
		matchers.append(ExactTermMatcher(syn_dict, syn_dict_name))
		matchers.append(BagTermMatcher(syn_dict, syn_dict_name))
	searchers = []
	for matcher in matchers:
		searchers.append(MaxInvTextSearcher(matcher))

	json_list = sorted(get_file_list(input_folder, lambda p: p.endswith('.json')))
	for searcher in searchers:
		print(searcher.name, '==============================================')
		for json_path in json_list:
			print(json_path)
			field_info = json.load(open(json_path))
			field2text = {field: field_info[field]['RAW_TEXT'] for field in field_info}
			field_info = handle_single(field2text, searcher, hpo2text)
			output_json = os.path.join(output_folder, searcher.name, '{}'.format(os.path.split(json_path)[1]))
			os.makedirs(os.path.dirname(output_json), exist_ok=True)
			json.dump(field_info, open(output_json, 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
	input_folder = os.path.join(DATA_PATH, 'raw', 'PUMC', 'case87-doc-hy-strict-enhance')
	output_folder = os.path.join(DATA_PATH, 'preprocess', 'patient', 'CCRD_OMIM_ORPHA', 'PUMC', 'case87-doc-hy-strict-enhance')
	process(input_folder, output_folder)




