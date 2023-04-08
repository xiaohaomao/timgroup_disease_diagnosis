import requests
from requests_toolbelt import MultipartEncoder
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
from core.utils.utils import get_logger, fill_dict_from_dict, strip_dict_key, list_add_tail
from core.utils.constant import LOG_PATH, DATA_PATH


logger = get_logger('DECIPHER_DOWNLOAD', log_path=LOG_PATH + '/DECIPHER_DOWNLOAD') # requests will output DEBUG INFO with the logger


def get_filter_for_gene(gene_name):
	"""
	Args:
		gene_name (str): e.g. 'SCN2A'
	Returns:
		str: e.g. '{"value":[{"field":"gene","value":"SCN2A","cmp":"overlaps","position":"2:166095912-166248818"}],"cmp":"and"}'
	"""
	r = requests.get('https://decipher.sanger.ac.uk/search?q={0}#consented-patients/results'.format(gene_name))
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	cp_div = soup.find(id='consented-patients')
	return cp_div['data-postdata']


def get_search_result_from_filter(filter_json):
	"""
	Args:
		filter_json (str): e.g. '{"value":[{"field":"gene","value":"SCN2A","cmp":"overlaps","position":"2:166095912-166248818"}],"cmp":"and"}'
	Returns:
		dict: {'SNV': [], 'CNV': []}
	"""
	m = MultipartEncoder(fields={'filter': filter_json})
	r = requests.post('https://decipher.sanger.ac.uk/search/consented-patients', data=m, headers={'Content-Type': m.content_type})
	r.raise_for_status()
	print(r.text)
	soup = BeautifulSoup(r.text, 'lxml')
	data_region = soup.find(lambda div: div.has_attr('data-region'))
	patient_dict = json.loads(data_region['data-features'])   # {'SNV': [], 'CNV': []}
	return patient_dict


def get_patient_num_from_filter(filter_json):
	"""
	Returns:
		int: patient number
	"""
	m = MultipartEncoder(fields={'filter': filter_json})
	r = requests.post('https://decipher.sanger.ac.uk/search/consented-patients/count', data=m, headers={'Content-Type': m.content_type})
	r.raise_for_status()
	return int(r.text)


def get_snv_from_patient_id(patientId):

	r = requests.get('https://decipher.sanger.ac.uk/patient/{0}/snvs'.format(patientId))
	r.raise_for_status()
	snv_list = []
	func_dict = {
		0: handle_snv_Td_location, 1: handle_td_gene_transcript, 2: handle_td_annotation,
		3: handle_td_inheritance_genotype, 4: handle_td_pathogenicity_contribution
	}
	soup = BeautifulSoup(r.text, 'lxml')
	snv_table = soup.find(id='patient-snvs-table')
	tbody = snv_table.find('tbody')

	for tr in tbody.find_all('tr'):
		snv_dict = {'SNV_ID': tr['data-hash'].split('/').pop()}
		tds = tr.find_all('td')
		for i in range(len(tds)):
			if i in func_dict:
				fill_dict_from_dict(snv_dict, func_dict[i](tds[i]))
		snv_list.append(strip_dict_key(snv_dict))
	return snv_list


def get_cnv_from_patient_id(patientId):
	"""
	Returns:
		list: e.g. patient 1414, [{
			'CNV_ID': '20990', 'LOCATION': '12:129147543-133777645', VARIANT_CLASS: 'Triplication', MEAN_RATIO: '1', SIZE: '4630103',
			'GENE_NUM': '54', 'INHERITANCE': 'Unknown', 'GENOTYPE': 'Heterozygous', 'PATHOGENICITY': '', 'CONTRIBUTION': ''
			'DS_SCORE': '14.76', 'SAMPLING_PROB': '<1 %'
		}]
	"""
	r = requests.get('https://decipher.sanger.ac.uk/patient/{0}/cnvs'.format(patientId))
	r.raise_for_status()
	cnv_list = []
	func_dict = {
		0: handle_td_location, 1: handle_td_class_mean_ratio, 2: handle_td_size, 3: handle_td_genes,
		4: handle_td_inheritance_genotype, 5: handle_td_pathogenicity_contribution, 6: handle_td_score_samplingprob
	}
	soup = BeautifulSoup(r.text, 'lxml')
	cnv_table = soup.find(id='patient-cnvs-table')
	tbody = cnv_table.find('tbody')
	for tr in tbody.find_all('tr'):
		cnv_dict = {'CNV_ID': tr['data-hash'].split('/').pop()}
		tds = tr.find_all('td')
		for i in range(len(tds)):
			if i in func_dict:
				fill_dict_from_dict(cnv_dict, func_dict[i](tds[i]))
		cnv_list.append(strip_dict_key(cnv_dict))
	return cnv_list


def handle_td_location(td):
	return {'LOCATION': td['data-order']}


def handle_snv_Td_location(td):
	ret_dict = handle_td_location(td)
	alleles_span = td.find('span', attrs={'class': 'alleles short'})
	ret_dict['ALLELES'] = alleles_span.get_text() if alleles_span else ''
	return ret_dict


def handle_td_class_mean_ratio(td):
	info = [text for text in td.stripped_strings]
	info = list_add_tail(info, '', 2-len(info))
	return {'VARIANT_CLASS': info[0], 'MEAN_RATIO': info[1]}


def handle_td_size(td):
	return {'SIZE': td['data-order']}


def handle_td_genes(td):
	return {'GENE_NUM': td.get_text()}


def handle_td_inheritance_genotype(td):
	info = [text for text in td.stripped_strings]
	info = list_add_tail(info, '', 2-len(info))
	return {'INHERITANCE': info[0], 'GENOTYPE': info[1]}


def handle_td_pathogenicity_contribution(td):
	info = [text for text in td.stripped_strings]
	info = list_add_tail(info, '', 2-len(info))
	return {'PATHOGENICITY': info[0], 'CONTRIBUTION': info[1]}


def handle_td_score_samplingprob(td):
	info = [a.get_text() for a in td.find_all('a')]
	info = list_add_tail(info, '', 2-len(info))
	return {'DS_SCORE': info[0], 'SAMPLING_PROB': info[1]}


def handle_td_gene_transcript(td):
	info = [text for text in td.stripped_strings]
	info = list_add_tail(info, '', 3-len(info))
	return {'GENE_NAME': info[0], 'TRANSCRIPT_1': info[1], 'TRANSCRIPT_2': info[2]}


def handle_td_annotation(td):

	info = [text for text in td.stripped_strings]
	print(info)
	info = list_add_tail(info, '', 2-len(info))
	return {'ANNOTATION_1': info[0], 'ANNOTATION_2': info[1]}


def get_phenotype_from_patient_id(patientId):
	"""
	Args:
		patientId (str or int)
	Returns:
		list: [HPO1, HPO2, ...], list of phenotypes
	"""
	def get_patient_dict():
		for pa_dict in data_dict['family']['people']:
			if pa_dict['relation'] == 'patient':
				assert str(pa_dict['patient_id']) == str(patientId)
				return pa_dict
		return {}
	r = requests.get('https://decipher.sanger.ac.uk/patient/{0}/phenotype'.format(patientId))
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	phe_div = soup.find(id='phenotype-lists')
	data_dict = json.loads(phe_div['data-state'])
	patient_dict = get_patient_dict()
	hpo_list = [pheDict['code'] for pheDict in patient_dict['phenotypes']]
	return hpo_list


def get_overview_from_patient_id(patientId):
	"""
	Args:
		patientId (str or int)
	Returns:
		list: [('Age at last clinical assessment': 'unknown'), ('Chromosomal sex': '46XY'), ('Open-access consent': 'Yes'), ...]
	"""
	r = requests.get('https://decipher.sanger.ac.uk/patient/{0}/overview'.format(patientId))
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	tbody = soup.find('tbody')
	if not tbody:
		return []
	data_list = []
	for tr in tbody.find_all('tr'):
		tds = tr.find_all('td')
		assert len(tds) == 2
		data_list.append([td.get_text(strip=True) for td in tds])
	return data_list


def get_genes_for_cnv(patientId, cnv_id):
	"""
	Args:
		patientId (str or int)
		cnv_id (str or int)
	Returns:
		list: [geneInfoDict, ...]
	"""
	r = requests.get('https://decipher.sanger.ac.uk/patient/{0}/cnv/{1}/genes'.format(patientId, cnv_id))
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	data_list = json.loads(soup.find('div', attrs={'data-component': 'genes'})['data-rows'])
	return data_list


ALL_PATIENT_ID_JSON = DATA_PATH + '/raw/DICIPHER/all_hpo_patients.json'
def get_all_patient_ids_with_hpo():
	if os.path.exists(ALL_PATIENT_ID_JSON):
		return json.load(open(ALL_PATIENT_ID_JSON))
	m = MultipartEncoder(fields={'filter': '{"value":[{"cmp":"like","value":"All","field":"phenotype"}],"cmp":"and"}'})

	r = requests.post('https://decipher.sanger.ac.uk/search/consented-patients', data=m, headers={'Content-Type': m.content_type})
	r.raise_for_status()
	soup = BeautifulSoup(r.text, 'lxml')
	tbody = soup.find(id='search-patients-table').find('tbody')
	patient_ids = []
	for tr in tbody.find_all('tr'):
		patient_ids.append(tr['data-href'].split('/').pop().strip())
	patient_ids = sorted([int(id) for id in patient_ids])
	patient_ids = [str(id) for id in patient_ids]
	json.dump(patient_ids, open(ALL_PATIENT_ID_JSON, 'w'), indent=2)
	return patient_ids


ALL_PATIENT_OVERVIEWS = DATA_PATH + '/raw/DICIPHER/all_patient_overviews.json'
def download_all_overviews(patient_ids):
	"""{patientId: info_list}, info_list=get_overview_from_patient_id(patientId)
	"""
	overviews = []
	with Pool() as pool:
		for overview in tqdm(pool.imap(get_overview_from_patient_id, patient_ids), total=len(patient_ids), leave=False):
			overviews.append(overview)
	data_dict = {patient_ids[i]: overviews[i] for i in range(len(patient_ids))}
	json.dump(data_dict, open(ALL_PATIENT_OVERVIEWS, 'w'), indent=2)


ALL_PATIENT_PHENOTYPES = DATA_PATH + '/raw/DICIPHER/all_patient_phenotypes.json'
def download_all_phenotypes(patient_ids):
	"""{patientId: [HPO1, HPO2, ...]}
	"""
	phenotypes = []
	with Pool() as pool:
		for hpo_list in tqdm(pool.imap(get_phenotype_from_patient_id, patient_ids), total=len(patient_ids), leave=False):
			phenotypes.append(hpo_list)
	data_dict = {patient_ids[i]: phenotypes[i] for i in range(len(patient_ids))}
	json.dump(data_dict, open(ALL_PATIENT_PHENOTYPES, 'w'), indent=2)


ALL_PATIENT_SNVs = DATA_PATH + '/raw/DICIPHER/all_patient_snvs.json'
def download_all_snvs(patient_ids):
	"""{patient_ids: snv_list}, snv_dict = get_snv_from_patient_id(pid)
	"""
	snv_lists = []
	with Pool() as pool:
		for snv_list in tqdm(pool.imap(get_snv_from_patient_id, patient_ids), total=len(patient_ids), leave=False):
			snv_lists.append(snv_list)
	data_dict = {patient_ids[i]: snv_lists[i] for i in range(len(patient_ids))}
	json.dump(data_dict, open(ALL_PATIENT_SNVs, 'w'), indent=2)


ALL_PATIENT_CNVs = DATA_PATH + '/raw/DICIPHER/all_patient_cnvs.json'
def download_all_cnvs(patient_ids):
	"""{patient_ids: cnv_list}, cnv_dict = get_cnv_from_patient_id(pid)
	"""
	cnv_lists = []
	with Pool() as pool:
		for cnv_list in tqdm(pool.imap(get_cnv_from_patient_id, patient_ids), total=len(patient_ids), leave=False):
			cnv_lists.append(cnv_list)
	data_dict = {patient_ids[i]: cnv_lists[i] for i in range(len(patient_ids))}
	json.dump(data_dict, open(ALL_PATIENT_CNVs, 'w'), indent=2)


def multi_wrap_get_genes_for_cnv(paras):
	pid, cnv_id = paras
	return pid, cnv_id, get_genes_for_cnv(pid, cnv_id)


ALL_CNV_DETAIL_GENEs = DATA_PATH + '/raw/DICIPHER/all_cnvs_detail_genes.json'
def download_all_genes_for_cnvs_detail():
	"""NOTE: be called after download_all_cnvs; {patientId: {cnv_id: gene_list}}
	"""
	cnv_data_dict = json.load(open(ALL_PATIENT_CNVs))
	data_dict = {}
	paras = [(patientId, cnv_dict['CNV_ID']) for patientId in cnv_data_dict for cnv_dict in cnv_data_dict[patientId]]
	with Pool() as pool:
		for pid, cnv_id, gene_list in tqdm(pool.imap(multi_wrap_get_genes_for_cnv, paras), total=len(paras), leave=False):
			if pid not in data_dict:
				data_dict[pid] = {}
			data_dict[pid][cnv_id] = gene_list
	json.dump(data_dict, open(ALL_CNV_DETAIL_GENEs, 'w'), indent=2)


ALL_CNV_GENEs = DATA_PATH + '/raw/DICIPHER/all_cnvs_genes.json'
def gen_all_genes_for_cnvs():
	data_dict = json.load(open(ALL_CNV_DETAIL_GENEs))
	for pid in data_dict:
		for cnv_id, gene_list in data_dict[pid].items():
			for i in range(len(gene_list)):
				omim_ids = ['OMIM:'+omim_dict['id'] for omim_dict in gene_list[i]['omim_id']] if gene_list[i]['omim_id'] else []
				for id in omim_ids:
					assert len(id) == 11
				gene_list[i] = {'OMIM_ID': omim_ids, 'GENE_NAME': gene_list[i]['name']}
	json.dump(data_dict, open(ALL_CNV_GENEs, 'w'), indent=2)


def statics(logger):
	from collections import Counter
	patient_ids = get_all_patient_ids_with_hpo()
	logger.info('patients with HPO: {0}'.format(len(patient_ids)))
	snv_data_dict = json.load(open(ALL_PATIENT_SNVs))
	logger.info('patients with HPO and SNV: {0}'.format(sum(map(lambda snv_list: len(snv_list)>0, snv_data_dict.values()))))
	logger.info('patients with HPO and single SNV: {0}'.format(sum(map(lambda snv_list: len(snv_list)==1, snv_data_dict.values()))))
	cnv_data_dict = json.load(open(ALL_PATIENT_CNVs))
	logger.info('patients with HPO and CNV: {0}'.format(sum(map(lambda cnv_list: len(cnv_list)>0, cnv_data_dict.values()))))
	logger.info('patients with HPO and single CNV: {0}'.format(sum(map(lambda cnv_list: len(cnv_list)==1, cnv_data_dict.values()))))
	phe_data_dict = json.load(open(ALL_PATIENT_PHENOTYPES))
	counter = Counter()
	for pid, hpo_list in phe_data_dict.items():
		counter[len(hpo_list)] += 1
	for hpo_num, count in sorted(counter.items()):
		logger.info('patients with {0} HPO Terms: {1}, {2}%'.format(hpo_num, count, 100*count/len(phe_data_dict)))
	logger.info('Average HPO Number per patient: {0}'.format(sum(map(lambda item: item[0]*item[1], counter.items()))/len(phe_data_dict)))


def test():
	pass


if __name__ == '__main__':
	patient_ids = get_all_patient_ids_with_hpo()
	download_all_phenotypes(patient_ids)
	download_all_snvs(patient_ids)
	download_all_cnvs(patient_ids)
	download_all_genes_for_cnvs_detail()
	gen_all_genes_for_cnvs()
	download_all_overviews(patient_ids)
	statics(logger)



