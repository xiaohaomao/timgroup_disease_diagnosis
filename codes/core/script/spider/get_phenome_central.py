

import requests
import json
import os
from core.utils.constant import DATA_PATH
from tqdm import tqdm

PUBLIC_PATIENT_JSON = DATA_PATH+'/raw/PhenomeCentral/phenotips_2018-09-14_01-10.json'
SIMILAR_CASE_FOLDER = DATA_PATH+'/raw/PhenomeCentral/SimilarCase'
cookie_str = 'username="N122xoG2BAjbsWe9GLs7xg__"; password="o0/92na58j44jFjew50Q4g__"; rememberme="false"; validation="b53d54ed41de86566ebeb6914763cbb2"; _ga=GA1.2.1133544449.1536901777; _gid=GA1.2.154615662.1536901777; JSESSIONID=5D827BFAEEFFA668774BEF1DCF18CA28; _gat=1'

def cookie_str_to_dict(cookie_str):

	return dict([line.strip().split('=', 1) for line in cookie_str.split(';')])


def get_similar_case(pid):
	data = {
		'query': pid,
		'outputSyntax': 'plain',
		'maxResults': 1000,
		'offset': 1
	}
	r = requests.post('https://phenomecentral.org/rest/patients/{0}/similar-cases'.format(pid), data=data, cookies=cookie_str_to_dict(cookie_str))
	r.raise_for_status()
	info_dict = json.loads(r.text)
	return info_dict


def download_all_similar_case():
	os.makedirs(SIMILAR_CASE_FOLDER, exist_ok=True)
	public_pa_info = json.load(open(PUBLIC_PATIENT_JSON))
	public_pids = [info_dict['report_id'] for info_dict in public_pa_info]
	for pid in tqdm(public_pids):
		similar_case_dict = get_similar_case(pid)
		json.dump(similar_case_dict, open(SIMILAR_CASE_FOLDER+'/{0}.json'.format(pid), 'w'), indent=2)


if __name__ == '__main__':

	download_all_similar_case()
	pass


