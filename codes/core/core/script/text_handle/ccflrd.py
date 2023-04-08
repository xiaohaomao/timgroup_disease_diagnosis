
# Compendium of China's First List of Rare Disease

import os

from core.text_handler.max_inv_text_searcher import MaxInvTextSearcher
from core.text_handler.term_matcher import BagTermMatcher
from core.utils.constant import RESULT_PATH, DATA_PATH
from core.text_handler.syn_generator import UMLSSynGenerator
from core.explainer.search_explainer import SearchExplainer


# def laron_syndrome():
# 	text = open(DATA_PATH+'/raw/MER/PUMC/莱伦综合征/莱伦综合征.txt').read()
# 	sg = UMLSSynGenerator()
# 	matcher = BagTermMatcher(sg.get_hpo_to_syn_terms_with_bg_evaluate(), 'BGEvaSynDict')
# 	searcher = MaxInvTextSearcher(matcher)
# 	search_result = searcher.search(text)
# 	output_txt = RESULT_PATH + '/text_handle/莱伦综合征/{}.txt'.format(searcher.name); os.makedirs(os.path.split(output_txt)[0], exist_ok=True)
# 	SearchExplainer([text], [search_result]).explain_save_txt(output_txt)

if __name__ == '__main__':
	pass
	laron_syndrome()



