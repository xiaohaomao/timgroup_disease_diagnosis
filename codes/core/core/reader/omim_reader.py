import os
import json
from bs4 import BeautifulSoup
import re
from core.utils.constant import DATA_PATH, TEMP_PATH, JSON_FILE_FORMAT
from core.utils.utils import set_if_not_empty, del_if_empty, read_standard_file, check_load_save

class OMIMReader(object):
	def __init__(self):
		PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'OMIM')
		os.makedirs(PREPROCESS_FOLDER, exist_ok=True)

		self.CNS_OMIM_HTML_PATH = os.path.join(DATA_PATH, 'raw', 'OMIM', 'OMIM_CHPO.htm')
		self.CNS_OMIM_JSON_PATH = os.path.join(PREPROCESS_FOLDER, 'cns_omim.json')
		self.cns_omim = None   # {code: {'CNS_NAME': '', 'ENG_NAME': ''}}
		self.OMIM_TITLE_TXT_PATH = os.path.join(DATA_PATH, 'raw', 'OMIM', '2017', 'mimTitles.txt')
		self.OMIM_JSON_PATH = os.path.join(PREPROCESS_FOLDER, 'omim.json')
		self.omim_dict = None    # {code: {'PREFIX': '', 'ENG_NAME': '', 'ALT_NAME': '', 'SYMBOL': ''}}
		self.OMIM_TO_GENE_TXT_PATH = os.path.join(DATA_PATH, 'raw', 'OMIM', '2017', 'mim2gene.txt')
		self.omim_to_gene_dict = None  # {code: {'TYPE': 'gene/phenotype', 'ENTREZ_ID': '217', 'GENE_SYMBOL': 'ALDH2', 'ENSEMBL_ID': 'ENSG00000111275'}}
		self.GENE_SYMBOL_MATCH_OMIM_ID_JSON = os.path.join(PREPROCESS_FOLDER, 'gene_symbol_match_omim_id.json')
		self.gene_symbol_to_omim_dict = None    # {geneSymbol: OMIM_ID}
		self.GENE_MAP_TXT = os.path.join(DATA_PATH, 'raw', 'OMIM', '2017', 'genemap.txt')
		self.gene_map_dict = None # {code: {'SORT': '', 'MONTH': '', 'DAY': '', 'YEAR': '', 'CYTO_LOCATION': '', 'GENE_SYMBOLS': '', 'CONFIDENCE': '', ...}, ...}
		self.GENE_MAP_DIS_JSON = os.path.join(PREPROCESS_FOLDER, 'gene_map_dis.json')
		self.gene_map_dis_dict = {} # {geneOMIMID: [disOMIMID1, ...]}
		self.OLD2NEW_JSON = os.path.join(PREPROCESS_FOLDER, 'old2new.json')


	def add_omim_prefix(self, omimNum):
		return 'OMIM:'+omimNum


	@check_load_save('cns_omim', 'CNS_OMIM_JSON_PATH', JSON_FILE_FORMAT)
	def get_cns_omim(self):
		"""
		Returns:
			dict: {omim_code: {'CNS_NAME': str, 'ENG_NAME': str}}
		"""
		cns_omim = {}
		soup = BeautifulSoup(open(self.CNS_OMIM_HTML_PATH), 'lxml')
		table = soup.find_all('table')[0]
		for tr in table.find_all('tr')[1:]:
			tds = tr.find_all('td')
			code = self.add_omim_prefix(tds[0].string.strip())
			cns_omim[code] = {}
			set_if_not_empty(cns_omim[code], 'CNS_NAME', (tds[2].string or '').strip())
			set_if_not_empty(cns_omim[code], 'ENG_NAME', (tds[1].string or '').strip())
		return cns_omim


	@check_load_save('omim_dict', 'OMIM_JSON_PATH', JSON_FILE_FORMAT)
	def get_omim_dict(self):
		"""
		Returns:
			dict: {code: {'PREFIX': '', 'ENG_NAME': '', 'ALT_NAME': '', 'SYMBOL': ''}}
		"""
		omim_dict = {}
		col_names = ['PREFIX', 'CODE', 'ENG_NAME', 'ALT_NAME', 'SYMBOL']
		for values in read_standard_file(self.OMIM_TITLE_TXT_PATH):
			assert len(values) == 5
			items = {col_names[i]: values[i] for i in range(len(values))}
			code = self.add_omim_prefix(items['CODE'])
			del items['CODE']
			omim_dict[code] = del_if_empty(items)
		return omim_dict


	def read_mim_to_gene(self):
		"""
		Note:
			phenotype & Entrez Gene ID(%): confirmed mendelian phenotype or phenotypic locus; molecular basis is not known
			phenotype only(#): a descriptive entry; usually of a phenotype; not represent a unique locus
			gene(*): a gene
			predominantly phenotypes(): a description of a phenotype; mendelian basis not been clearly established; separateness unclear
			moved/removed(^): no longer exists
			gene/phenotype(+): contains the description of a gene of known sequence and a phenotype
		Returns:
			dict: {code: {'TYPE': 'gene/phenotype', 'ENTREZ_ID': '217', 'GENE_SYMBOL': 'ALDH2', 'ENSEMBL_ID': 'ENSG00000111275'}}
		"""
		if self.omim_to_gene_dict:
			return self.omim_to_gene_dict
		self.omim_to_gene_dict = {}
		col_names = ['OMIM_ID', 'TYPE', 'ENTREZ_ID', 'GENE_SYMBOL', 'ENSEMBL_ID']
		for values in read_standard_file(self.OMIM_TO_GENE_TXT_PATH):
			self.omim_to_gene_dict[self.add_omim_prefix(values[0])] = {col_names[i]: values[i] for i in range(1, len(values)) if values[i]}
		return self.omim_to_gene_dict


	def get_gene_symbol_match_id(self):
		"""
		Returns:
			dict: {gene_name: OMIM_ID}
		"""
		if self.gene_symbol_to_omim_dict:
			return self.gene_symbol_to_omim_dict
		if os.path.exists(self.GENE_SYMBOL_MATCH_OMIM_ID_JSON):
			self.gene_symbol_to_omim_dict = json.load(open(self.GENE_SYMBOL_MATCH_OMIM_ID_JSON))
			return self.gene_symbol_to_omim_dict
		self.gene_symbol_to_omim_dict = {}
		omim_to_gene_dict = self.read_mim_to_gene()
		for omim_id, info_dict in omim_to_gene_dict.items():
			if 'GENE_SYMBOL' in info_dict:
				self.gene_symbol_to_omim_dict[info_dict['GENE_SYMBOL']] = omim_id
		json.dump(self.gene_symbol_to_omim_dict, open(self.GENE_SYMBOL_MATCH_OMIM_ID_JSON, 'w'), indent=2)
		return self.gene_symbol_to_omim_dict


	def read_gene_map(self):
		"""
		Returns:
			dict: {omim_id: info_dict}, info_dict = {'SORT': '', 'MONTH': '', ...}
		"""
		if self.gene_map_dict:
			return self.gene_map_dict
		self.gene_map_dict = {}
		col_names = [
			'SORT', 'MONTH', 'DAY', 'YEAR', 'CYTO_LOCATION', 'GENE_SYMBOLS', 'CONFIDENCE', 'GENE_NAME', 'MIM_NUMBER',
			'MAPPING_METHOD', 'COMMENTS', 'PHENOTYPES', 'MOUSE_GENE_SYMBOL'
		]
		col_num = len(col_names)
		for values in read_standard_file(self.GENE_MAP_TXT):
			assert len(values) == col_num
			info_dict = {col_names[i]: values[i] for i in range(len(values)) if values[i]}
			omim_id = info_dict['MIM_NUMBER']
			self.gene_map_dict[self.add_omim_prefix(omim_id)] = info_dict
		return self.gene_map_dict


	def get_gene_match_disease(self):

		if self.gene_map_dis_dict:
			return self.gene_map_dis_dict
		if os.path.exists(self.GENE_MAP_DIS_JSON):
			self.gene_map_dis_dict = json.load(open(self.GENE_MAP_DIS_JSON))
			return self.gene_map_dis_dict
		self.gene_map_dis_dict = {}
		p_str = ', (\d{6}) \('
		p = re.compile(p_str)
		gene_map_dict = self.read_gene_map()
		for omim_id, info_dict in gene_map_dict.items():
			dis_text = info_dict.get('PHENOTYPES', '')
			id_list = [self.add_omim_prefix(id) for id in p.findall(dis_text)]
			if id_list:
				self.gene_map_dis_dict[omim_id] = id_list
		json.dump(self.gene_map_dis_dict, open(self.GENE_MAP_DIS_JSON, 'w'), indent=2)
		return self.gene_map_dis_dict


	def get_old_to_new_omim(self, omim_codes):

		pass


if __name__ == '__main__':

	pass