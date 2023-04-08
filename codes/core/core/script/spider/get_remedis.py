import re
import requests
from requests_toolbelt import MultipartEncoder  # multipart/form-data
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

from core.utils.utils import get_logger, fill_dict_from_dict, strip_dict_key, list_add_tail, load_save_for_func, unique_list
from core.utils.constant import LOG_PATH, DATA_PATH, JSON_FILE_FORMAT

OUTPUT_FOLDER = DATA_PATH + '/raw/RAMEDIS'; os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_all_author():
	return {
		'53': 'Riyadh Armed Forces Hospital - Abdulrahman Alswaid',
		'4': 'Published Case - Published Cases',
		'41': 'University Childrens Hospital Heidelberg - Dietrich Feist',
		'14': 'Childrens Hospital Reutlingen - G. Frauendienst-Egger',
		'11': 'University Childrens Hospital Heidelberg - Dorothea Haas',
		'73': 'University Childrens Hospital Erlangen - Ina Knerr',
		'32': 'University Childrens Hospital Muenster - Hans Georg Koch',
		'3': 'Zentrum fuer Stoffwechseldiagnostik Reutlingen - Herbert Korall',
		'7': 'University Childrens Hospital Freiburg - Willy Lehnert',
		'33': 'University Childrens Hospital Muenster - Christina Leyendecker',
		'1': 'Bielefeld University - Thoralf Toepel',
		'2': 'Childrens Hospital Reutlingen - Friedrich Trefz'
	}


def get_author_to_von_pids(author_id, von):
	"""
	Returns:
		list: [pid1, pid2, ...]
	"""
	print('get_author_to_von_pids: author_id={}, von={}'.format(author_id, von))
	r = requests.get('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/searchv2/search_author_ok.php', {'id':author_id, 'von':von})
	r.raise_for_status()
	pid_list = re.findall('A HREF="\.\./dm-guest/case_main\.php\?pat_id=(\d+)"', r.text)
	return pid_list


def get_author_to_pids(author_id):
	pid_list = []
	von, step = 0, 10
	while True:
		von_pid_list = get_author_to_von_pids(author_id, von)
		if len(von_pid_list) == 0:
			break
		pid_list.extend(von_pid_list)
		von += step
	return pid_list


@load_save_for_func(OUTPUT_FOLDER+'/author_to_pids.json', JSON_FILE_FORMAT)
def get_all_author_to_pids():
	"""
	Returns:
		dict: {author_id: [pid1, ...]}
	"""
	author_to_name = get_all_author()
	return {author_id: get_author_to_pids(author_id) for author_id, _ in author_to_name.items()}


def get_all_pids():
	author_to_pids = get_all_author_to_pids()
	return unique_list([pid for pid_list in author_to_pids.values() for pid in pid_list])


def get_main_data(pid, sess):
	r = sess.get('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_main.php?pat_id={}'.format(pid))
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	ret_dict = {
		'Patient ID': soup.find('td', text='Patient ID').find_next_sibling('td').get_text(strip=True),
		'Diagnosis': soup.find('td', text='Diagnosis').find_next_sibling('td').get_text(strip=True),
	}
	ret_dict.update(column_to_dict(soup.find('td', text='Gender').find_parent('table')))
	ret_dict.update(column_to_dict(soup.find('td', text='Author').find_parent('table')))
	return ret_dict


def column_to_dict(table, kcol=0, vcol=1):
	"""
	Returns:
		dict: {k: v}
	"""
	ret_dict = {}
	for tr in table.find_all('tr'):
		tds = tr.find_all('td')
		if len(tds) <= vcol:
			continue
		ret_dict[tds[kcol].get_text(strip=True)] = tds[vcol].get_text(strip=True)
	return ret_dict


def get_molecular_genetics(sess):
	r = sess.get('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_genetics.php')
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	ret_dict = {
		'Gene name': soup.find('td', text='Gene name').find_next_sibling('td').get_text(strip=True),
		'Genotype':soup.find('td', text='Genotype').find_next_sibling('td').get_text(strip=True),
	}
	for trivialTd in soup.find_all('td', text='Trivial name'):
		table = trivialTd.find_parent('table')
		allele = table.find_parent('tr').find_previous_sibling('tr').get_text(strip=True)
		ret_dict[allele] = column_to_dict(table)
	return ret_dict


def get_LabFindings(sess):
	return get_sub_report('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_lab.php', sess)


def get_symptoms(sess):
	return get_sub_report('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_symptoms.php', sess)


def get_diet_drugs(sess):
	return get_sub_report('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_dietdrugs.php', sess)


def get_therapy(sess):
	return get_sub_report('https://agbi.techfak.uni-bielefeld.de/ramedis/htdocs/dm-guest/case_therapy.php', sess)


def get_sub_report(url, sess):
	r = sess.get(url)
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	table = soup.find('td', text='Name').find_parent('table')
	return table_to_row_item_list(table)


def table_to_row_item_list(table):
	"""
	Returns:
		list: [{k1: v, k2: v}, ...]
	"""
	ret_list = []
	trs = table.find_all('tr')
	keys = [td.get_text(strip=True) for td in trs[0].find_all('td')]
	for tr in trs[1:]:
		values = [td.get_text(strip=True) for td in tr.find_all('td')]
		ret_list.append({k: v for k, v in zip(keys, values)})
	return ret_list


def get_patient(pid):
	output_json = OUTPUT_FOLDER + '/patients/{}.json'.format(pid)
	if os.path.exists(output_json):
		return json.load(open(output_json))
	sess = requests.Session()
	patient_dict = {
		'MainData': get_main_data(pid, sess),
		'Molecular genetics': get_molecular_genetics(sess),
		'Lab findings': get_LabFindings(sess),
		'Symptoms': get_symptoms(sess),
		'Diet/drugs': get_diet_drugs(sess),
		'Therapy/development': get_therapy(sess),
	}
	json.dump(patient_dict, open(output_json, 'w'), indent=2, ensure_ascii=False)
	return patient_dict


def get_patient_wrapper(pid):
	return get_patient(pid), pid


@load_save_for_func(OUTPUT_FOLDER+'/ramedis_patients.json', JSON_FILE_FORMAT)
def spider():
	"""
	Returns:
		dict: {
			pid: patientIem
		}
	"""
	pid_list = get_all_pids()
	return {pid: get_patient(pid) for pid in tqdm(pid_list)}


@load_save_for_func(OUTPUT_FOLDER+'/ramedis_patients.json', JSON_FILE_FORMAT)
def spider_parallel():
	ret_dict = {}
	pid_list = get_all_pids()
	with Pool(12) as pool:
		for pa_dict, pid in tqdm(pool.imap_unordered(get_patient_wrapper, pid_list), total=len(pid_list), leave=False):
			ret_dict[pid] = pa_dict
	return ret_dict


if __name__ == '__main__':
	pass


	get_all_author_to_pids()
	spider_parallel()

