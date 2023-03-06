

import requests
import re
import os
import json
from tqdm import tqdm

from core.utils.constant import DATA_PATH, VALIDATION_TEST_DATA
from core.utils.utils import timer
from core.reader.hpo_filter_reader import HPOFilterDatasetReader

cookie_str = 'donationPopupEpoch=1585910722; _ga=GA1.2.779164720.1585910722; csrftoken=SvbVAgJBSN2CgL6eTmRxj5S58dD87Zt1rxPoIXhlTEBKAFLiJW3mmVY65jA71Ltb; _gid=GA1.2.925510869.1587363732; sessionid=9ebuatwiv1cc0cv70er7g7rj2svd8m4r; _gat=1'

headers = {
	'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
	'cookie': cookie_str,
	'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
	'accept-encoding': 'gzip, deflate, br',
	'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
	'cache-control': 'no-cache',
	'dnt': '1',
	'pragma': 'no-cache',
	'referer': 'https://omim.org/',
	'sec-fetch-dest': 'document',
	'sec-fetch-mode': 'navigate',
	'sec-fetch-site': 'same-origin',
	'sec-fetch-user': '?1',
	'upgrade-insecure-requests': '1'
}

def get_new_code(old_code):
	assert old_code.startswith('OMIM:')
	id = old_code.split(':').pop()
	r = requests.get('https://omim.org/entry/{}'.format(id), {
		'search': id, 'highlight': id},
		headers=headers
	)
	r.raise_for_status()
	search_obj = re.search('- \^? (\d+) - MOVED TO (\d+)', r.text)
	if search_obj:
		assert search_obj.group(1) == id
		return 'OMIM:'+search_obj.group(2)
	return None


def get_and_save_new_codes(old_codes, save_json):
	if os.path.exists(save_json):
		old2new = json.load(open(save_json))
	else:
		old2new = {}
	for i, old_code in tqdm(list(enumerate(old_codes))):
		if old_code in old2new:
			continue
		old2new[old_code] = get_new_code(old_code)
		if i % 100 == 0:
			json.dump(old2new, open(save_json, 'w'), indent=2)
	json.dump(old2new, open(save_json, 'w'), indent=2)
	return old2new


def get_hpo_reader(keep_dnames=None):
	keep_dnames = keep_dnames or ['OMIM', 'ORPHA', 'CCRD']
	return HPOFilterDatasetReader(keep_dnames=keep_dnames)


@timer
def get_phenomizer_omim_codes():
	from core.predict.model_testor import ModelTestor
	raw_results, patients = ModelTestor(hpo_reader=get_hpo_reader()).get_phenomizer_raw_results_and_patients('RAMEDIS_SAMPLE_30')
	return sorted([dis_code for dis_code, score in raw_results[0] if dis_code.startswith('OMIM:')])


@timer
def get_dataset_omim_codes():
	from core.helper.data.data_helper import DataHelper
	dh = DataHelper(hpo_reader=get_hpo_reader())
	omim_codes = []
	for data_name in ['RAMEDIS', 'MME', 'CJFH', 'SIM_ORIGIN']:
		patients = json.load(open(dh.origin_to_path[data_name]))
		omim_codes.extend([dis_code for hpo_list, dis_codes in patients for dis_code in dis_codes if dis_code.startswith('OMIM')])
	patients = dh.get_dataset('PUMC', VALIDATION_TEST_DATA, filter=False)
	omim_codes.extend([dis_code for hpo_list, dis_codes in patients for dis_code in dis_codes if dis_code.startswith('OMIM')])
	return list(set(omim_codes))


@timer
def get_anno_omim_codes():
	return get_hpo_reader(['OMIM']).get_dis_list()


def run():
	def check(omim_codes):
		for omim_code in omim_codes:
			assert omim_code.startswith('OMIM:')
	save_json = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'OMIM', 'old2new.json')
	omim_codes = []
	omim_codes.extend(get_phenomizer_omim_codes()); check(omim_codes)
	omim_codes.extend(get_dataset_omim_codes()); check(omim_codes)
	omim_codes.extend(get_anno_omim_codes()); check(omim_codes)
	omim_codes = sorted(list(set(omim_codes)))
	get_and_save_new_codes(omim_codes, save_json)


if __name__ == '__main__':


	run()


