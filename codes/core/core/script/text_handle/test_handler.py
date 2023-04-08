import os
from tqdm import tqdm
import json
from copy import deepcopy

from core.text_handler.max_inv_text_searcher import MaxInvTextSearcher
from core.text_handler.text_searcher import TextSearcher
from core.text_handler.term_matcher import ExactTermMatcher, BagTermMatcher
from core.reader.hpo_reader import HPOReader
from core.utils.constant import RESULT_PATH, DATA_PATH
from core.patient.gu_patient_generator import GuPatientGenerator
from core.text_handler.syn_generator import UMLSSynGenerator
from core.explainer.search_explainer import SearchExplainer


def handle_text_list(text_list, folder_path, gold_match=None):
	os.makedirs(folder_path, exist_ok=True)
	sg = UMLSSynGenerator()
	all_syn_dict = {
		'SimSynDict': sg.get_hpo_to_syn_terms(),
		'BGEvaSynDict': sg.get_hpo_to_syn_terms_with_bg_evaluate(),
		'SourceSynDict': sg.get_hpo_to_source_syn_terms(),
	}
	matchers = []
	for syn_dict_name, syn_dict in all_syn_dict.items():
		matchers.append(ExactTermMatcher(syn_dict, syn_dict_name, gold_match))
		matchers.append(BagTermMatcher(syn_dict, syn_dict_name, gold_match))
	searchers = []
	for matcher in matchers:
		searchers.append(MaxInvTextSearcher(matcher))

	for searcher in searchers:
		output_txt = os.path.join(folder_path, '{}.txt'.format(searcher.name))
		results = [searcher.search(p_text) for p_text in text_list]
		SearchExplainer(text_list, results).explain_save_txt(output_txt)


# def jym():
# 	patient_json = DATA_PATH + '/raw/MER/PUMC/贾耀敏/贾耀敏.json'
# 	p_text = '\n'.join(json.load(open(patient_json)).values())
# 	folder_path = RESULT_PATH + '/text_handle/贾耀敏'
# 	handle_text_list([p_text], folder_path)


if __name__ == '__main__':
	# jym()
	pass









