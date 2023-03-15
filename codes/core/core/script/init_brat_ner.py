

import json

from core.reader.hpo_reader import HPOReader
from core.utils.utils import get_all_descendents_for_many


def init_hpo_dict():
	HPO_JSON_PATH = 'hpo.json'
	hpo_reader = HPOReader()
	chpo_dict = hpo_reader.get_chpo_dict()
	hpo_dict = hpo_reader.get_hpo_dict()
	code2term = {}
	for hpo in hpo_dict:
		term = chpo_dict.get(hpo, {}).get('CNS_NAME', '') or hpo_dict[hpo].get('ENG_NAME', '')
		code2term[hpo] = term
	json.dump(code2term, open(HPO_JSON_PATH, 'w'), ensure_ascii=False, indent=2)


def gen_txt(hpo_list, path):
	hpo_reader = HPOReader()
	chpo_dict = hpo_reader.get_chpo_dict()
	hpo_set = get_all_descendents_for_many(hpo_list, hpo_reader.get_slice_hpo_dict())
	lines = []
	for hpo in hpo_set:
		if hpo in chpo_dict:
			lines.append('{} | {} \n'.format(hpo, chpo_dict[hpo]['CNS_NAME']))
	with open(path, 'w') as f:
		f.writelines(lines)


def init_hpo_include_dict():
	gen_txt(['HP:0000118', 'HP:0000005'], 'hpo_include.txt')


def init_hpo_exclude_dict():
	gen_txt(['HP:0012823', 'HP:0040006', 'HP:0040279'], 'hpo_exclude.txt')


if __name__ == '__main__':
	init_hpo_dict()
	init_hpo_include_dict()
	init_hpo_exclude_dict()

