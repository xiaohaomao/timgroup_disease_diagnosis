

import os
import itertools
import json

from bert_syn.core.data_helper import SynDictReader, HPOReader
from bert_syn.core.baseline import DictSim
from bert_syn.core.model_testor import ModelTestor
from bert_syn.utils.utils import timer
from bert_syn.utils.constant import DATA_PATH

@timer
def process_hpo_to_syn_terms(hpo_to_syn_terms):
	ret_dict = {}
	for hpo, syn_terms in hpo_to_syn_terms.items():
		syn_terms = list(set([t.strip() for t in syn_terms]))
		if syn_terms:
			ret_dict[hpo] = syn_terms
	return ret_dict


@timer
def run_predict(syn_dict_name, match_type, predict_terms_json, predict_raw_to_true_json):
	if syn_dict_name == 'chpo':
		hpo_to_cns = HPOReader().get_hpo_to_cns()
		hpo_to_syn_terms = {hpo: [term] for hpo, term in hpo_to_cns.items()}
	elif syn_dict_name == 'chpo_umls_source':
		hpo_to_syn_terms = SynDictReader().get_hpo_to_source_syn_terms()
	elif syn_dict_name == 'chpo_umls_bg_source':
		hpo_to_syn_terms = SynDictReader().get_hpo_to_syn_terms_with_bg_evaluate()
	else:
		raise RuntimeError('Unknown syn dict name: {}'.format(syn_dict_name))
	hpo_to_syn_terms = process_hpo_to_syn_terms(hpo_to_syn_terms)
	dict_sim = DictSim(f'{syn_dict_name}-{match_type}', hpo_to_syn_terms, match_type)

	samples = dict_sim.predict(predict_terms_json, cpu_use=12, chunksize=50)
	mt = ModelTestor()
	metric_dict, raw_term_to_rank = mt.cal_metrics(samples, json.load(open(predict_raw_to_true_json)), cpu_use=12, chunksize=10)

	prefix, postfix = os.path.splitext(predict_terms_json.replace(os.path.join(DATA_PATH, 'preprocess', 'dataset'), dict_sim.RESULT_SAVE_FOLDER))
	json.dump(
		sorted([(rank, term) for term, rank in raw_term_to_rank.items()]),
		open(prefix + f'-term-rank.json', 'w'), indent=2, ensure_ascii=False)
	json.dump(metric_dict, open(prefix + f'-metric.json', 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
	predict_terms_json = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'eval', 'ehr_terms_100.json')
	predict_raw_to_true_json = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', 'eval', 'ehr_to_true_texts_100.json')
	for syn_dict_name, match_type in itertools.product(['chpo', 'chpo_umls_source', 'chpo_umls_bg_source'], ['exact', 'bag', 'jaccard']):
		print('Running', syn_dict_name, match_type)
		run_predict(syn_dict_name, match_type, predict_terms_json, predict_raw_to_true_json)


