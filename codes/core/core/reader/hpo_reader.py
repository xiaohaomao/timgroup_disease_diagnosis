import json, pickle
import os
from tqdm import tqdm
import re
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from core.utils.constant import SPARSE_NPZ_FILE_FORMAT, JSON_FILE_FORMAT, PKL_FILE_FORMAT
from core.utils.constant import DATA_PATH, LOG_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, PHELIST_DESCENDENT, PHELIST_ORIGIN, PHELIST_ANCESTOR_DUP
from core.utils.utils import get_logger, dict_list_add, reverse_dict_list, delete_redundacy, read_standard_file, item_list_to_rank_list
from core.utils.utils import slice_dict_with_keep_set, check_load_save, check_load, get_all_ancestors_for_many, get_all_descendents_for_many, list_add_tail, get_all_dup_ancestors_for_many
from core.utils.utils import unique_list, get_all_descendents_with_dist, check_return, data_to_cooccur_matrix, data_to_01_matrix, get_all_ancestors
from core.utils.utils import dict_change_key_value
from core.reader.obo_reader import OBOReader


class HPOReader(object):

	def __init__(self):
		self.name = 'ALL'
		#self.name = 'OMIM'
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'HPO')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)

		self.CHPO_XLS_PATH = os.path.join(DATA_PATH, 'raw','CHPO', 'chpo.2016-10.xls')

		self.CHPO_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'chpo_dict.json')
		self.chpo_dict = None    # {CODE: {'CNS_NAME': .., 'ENG_NAME': ..}}

		self.CCRD_JSON_PATH = os.path.join(DATA_PATH, 'raw', 'CCRD', 'conpendium_hpo_process.json')
		self.HPO_OBO_PATH = os.path.join(DATA_PATH, 'raw', 'HPO', '2019', 'Ontology', 'hp.obo')
		self.HPO_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'hpo_dict.json')
		self.hpo_dict = None # {CODE: {'NAME': .., 'IS_A': [], 'CHILD': [], ...}}
		self.HPO_SLICE_JSON = os.path.join(self.PREPROCESS_FOLDER, 'hpo_slice_dict.json')
		self.slice_hpo_dict = None    #
		self.HPO_LIST_JSON = os.path.join(self.PREPROCESS_FOLDER, 'hpo_list.json')
		self.hpo_list = None # [hpo_code1, hpo_code2, ...]
		self.HPO_MAP_RANK_JSON = os.path.join(self.PREPROCESS_FOLDER, 'hpo_map_rank.json')
		self.hpo_map_rank = None  # {hpo_rank: hpo_code}
		self.USED_HPO_LIST_JSON = os.path.join(self.PREPROCESS_FOLDER, 'used_hpo_list.json')
		self.used_hpo_list = None
		self.HPO_INT_DICT_PKL = os.path.join(self.PREPROCESS_FOLDER, 'hpo_int_dict.pkl')
		self.hpo_int_dict = None  # {hpo_rank: {'IS_A': [hpoRank1, ...], 'CHILD': [hpoRank2, ...]}};

		self.OLDMAPNEWHPO_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'old_map_new_hpo.json')
		self.old_map_new_hpo = None # {OLD_HPO_CODE: NEW_HPO_CODE}


		self.ANNOTATION_TAB_PATH = os.path.join(DATA_PATH, 'raw', 'HPO', '2019', 'Annotations', 'phenotype_annotation.tab')
		self.ANNOTATION_HPOA_PATH = os.path.join(DATA_PATH, 'raw', 'HPO', '2019', 'Annotations', 'phenotype.hpoa')
		self.DISTOHPO_TAB_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'dis2hpo_tab.json')
		self.DISTOHPO_HPOA_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'dis2hpo_hpoa.json')
		self.dis2hpo = None    # {dis_code: [hpo_code, ...]}
		self.DISTOHPO_REDUCE_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'dis2hpo_reduce.json')
		self.dis2hpo_reduce = None  # {dis_code: [hpo_code, ...]}
		self.HPOTODIS_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'hpo2dis.json')
		self.hpo2dis = None    # {hpo_code: [dis_code, ...]}
		self.HPOTODIS_REDUCE_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'hpo2dis_reduce.json')
		self.hpo2dis_reduce = None  # {hpo_code: [dis_code, ...]}
		self.ANNO_HPO_JSON = os.path.join( self.PREPROCESS_FOLDER, 'anno_hpo.json')
		self.anno_hpo_list = None

		self.DIS_TO_HPO_PROB_TAB_JSON = os.path.join(self.PREPROCESS_FOLDER, 'dis_to_hpo_prob_tab.json')
		self.DIS_TO_HPO_PROB_HPOA_JSON = os.path.join(self.PREPROCESS_FOLDER, 'dis_to_hpo_prob_hpoa.json')
		self.DIS_TO_HPO_PROB_RAW_JSON = os.path.join(self.PREPROCESS_FOLDER, 'dis_to_hpo_prob_raw_hpoa.json')
		self.DIS_LIST_JSON = os.path.join(self.PREPROCESS_FOLDER, 'dis_list.json')
		self.dis_list = None # [disease_code1, disease_code2, ...]
		self.DIS_MAP_RANK_JSON = os.path.join(self.PREPROCESS_FOLDER, 'dis_map_rank.json')
		self.dis_map_rank = None  # {disease_rank: disease_code}
		self.dis_int_to_hpo_int = None  # {disRank: [hpo_rank, ...]}
		self.dis_int_to_hpo_int_reduce = None
		self.hpo_int_2_dis_int = None
		self.hpo_int_2_dis_int_reduce = None



		self.HPO_GENES_TO_DISEASES_TXT = os.path.join(DATA_PATH, 'raw', 'HPO', '2019', 'Genes', 'genes_to_diseases.txt')
		self.GENE_TO_SYMBOL_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene2symbol.json')
		self.gene2symbol = None
		self.GENE_TO_DIS_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene2dis.json')
		self.gene2dis = None # {gene_code: [dis1, dis2, ...]}

		self.GENE_TO_PHENOTYPE_TXT = os.path.join(DATA_PATH, 'raw', 'HPO', '2019', 'Annotations', 'ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt')
		self.GENETOHPO_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene2hpo.json')
		self.gene2hpo = None   # {gene_code: [hpo_code, ...]}
		self.GENETOHPO_REDUCE_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene2hpo_reduce.json')
		self.gene2hpo_reduce = None # {gene_code: [hpo_code, ...]}
		self.HPOTOGENE_JSON = os.path.join(DATA_PATH, self.PREPROCESS_FOLDER, 'hpo2gene.json')
		self.hpo2gene = None   # {hpo_code: [gene_code, ...]}
		self.HPOTOGENE_REDUCE_JSON = os.path.join(self.PREPROCESS_FOLDER, 'hpo2gene_reduce.json')
		self.hpo2gene_reduce = None

		self.GENE_LIST_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene_list.json')
		self.gene_list = None    # [geneCode1, ...]
		self.GENE_MAP_RANK_JSON = os.path.join(self.PREPROCESS_FOLDER, 'gene_map_rank.json')
		self.gene_map_rank = None # {gene_code: geneRank}
		self.gene_int_to_hpo_int = None
		self.gene_int_to_hpo_int_reduce = None
		self.hpo_int_to_gene_int = None
		self.hpo_int_to_gene_int_reduce = None

		self.hpo2freq_dict = None
		self.word2freq_dict = None

		self.ADJ_MAT_NPZ = os.path.join(self.PREPROCESS_FOLDER, 'hpo_adj_mat.npz')
		self.adj_mat = None
		self.PARENT_MAT_NPZ = os.path.join(self.PREPROCESS_FOLDER, 'hpo_parent_mat.npz')
		self.parent_mat = None
		self.hpo2depth = None

		self.dis_num = None
		self.hpo_num = None


	def split_dis_code(self, dis_code):
		return dis_code.split(':')


	@check_load_save('chpo_dict', 'CHPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_chpo_dict(self):

		import xlrd
		chpo_dict = {}

		col_names = ['MAIN_CLASS', 'CODE', 'ENG_NAME', 'CNS_NAME', 'ENG_DEF', 'CNS_DEF']

		save_keys = ['MAIN_CLASS', 'ENG_NAME', 'CNS_NAME', 'ENG_DEF', 'CNS_DEF']


		sheet = xlrd.open_workbook(self.CHPO_XLS_PATH).sheet_by_name('CHPO')


		for row in range(1, sheet.nrows):
			item = {col_names[col]:sheet.cell(row, col).value for col in range(sheet.ncols)}


			chpo_dict[item['CODE']] = {key: item[key] for key in save_keys if item[key]} # Attension: Value Not Empty


		chpo_dict['HP:0000118'] = {'CNS_NAME': '表型异常', 'ENG_NAME': 'Phenotypic abnormality',
								   'CNS_DEF': '', 'ENG_DEF': ''}
		old_to_new_hpo = self.get_old_map_new_hpo_dict()
		for hpo in list(chpo_dict.keys()):
			if hpo in old_to_new_hpo and old_to_new_hpo[hpo] not in chpo_dict:
				chpo_dict[old_to_new_hpo[hpo]] = chpo_dict[hpo]
				del chpo_dict[hpo]
		chpo_dict = slice_dict_with_keep_set(chpo_dict, set(self.get_hpo_list()))



		return chpo_dict


	@check_load_save('hpo_dict', 'HPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_hpo_dict(self):

		return OBOReader().load(self.HPO_OBO_PATH)


	@check_load_save('slice_hpo_dict', 'HPO_SLICE_JSON', JSON_FILE_FORMAT)
	def get_slice_hpo_dict(self):
		hpo_dict = self.get_hpo_dict()
		slice_hpo_dict = {code: slice_dict_with_keep_set(hpo_dict[code], {'IS_A', 'CHILD'}) for code in hpo_dict}
		return slice_hpo_dict


	def get_hpo_to_syn_list(self):
		"""
		Returns:
			dict: {hpo_code: [synTerm, ...]}
		"""
		hpo_dict = self.get_hpo_dict()
		return {hpo: unique_list([info['ENG_NAME']]+info.get('SYNONYM', []))  for hpo, info in hpo_dict.items()}


	def get_syn2hpo(self):
		"""
		Returns:
			dict: {synTerm: hpo}
		"""
		hpo_to_syn_list = self.get_hpo_to_syn_list()
		syn_to_hpo_list = reverse_dict_list(hpo_to_syn_list)
		for syn in syn_to_hpo_list:    # check
			if len(syn_to_hpo_list[syn]) > 1:
				print('Warning:', syn, syn_to_hpo_list[syn])
			syn_to_hpo_list[syn] = syn_to_hpo_list[syn][0]
		return syn_to_hpo_list


	@check_return('hpo_num')
	def get_hpo_num(self):
		return len(self.get_hpo_dict())


	@check_return('dis_num')
	def get_dis_num(self):
		return len(self.get_dis_to_hpo_dict())


	def get_gene_num(self):
		return len(self.get_gene_to_hpo_dict())


	def gen_rank_json(self, item_list, item_list_json, item_map_rank_json):

		item_num = len(item_list)
		item_map_rank = {item_list[i]: i for i in range(item_num)}
		json.dump(item_list, open(item_list_json, 'w'), indent=2)
		json.dump(item_map_rank, open(item_map_rank_json, 'w'), indent=2)
		return item_list, item_map_rank


	@check_load('hpo_list', 'HPO_LIST_JSON', JSON_FILE_FORMAT)
	def get_hpo_list(self):
		"""
		Returns:
			list: [hpo_code1, hpo_code2, ...]
		"""
		hpo_list, _ = self.gen_rank_json(list(self.get_hpo_dict().keys()), self.HPO_LIST_JSON, self.HPO_MAP_RANK_JSON)
		return hpo_list


	@check_load('hpo_map_rank', 'HPO_MAP_RANK_JSON', JSON_FILE_FORMAT)
	def get_hpo_map_rank(self):
		"""
		Returns:
			dict: {hpo_code: hpo_rank}
		"""
		_, hpo_map_rank = self.gen_rank_json(list(self.get_hpo_dict().keys()), self.HPO_LIST_JSON, self.HPO_MAP_RANK_JSON)
		return hpo_map_rank


	@check_load_save('used_hpo_list', 'USED_HPO_LIST_JSON', JSON_FILE_FORMAT)
	def get_used_hpo_list(self):
		dis_to_hpo = self.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
		return unique_list([hpo for hpo_list in dis_to_hpo.values() for hpo in hpo_list])


	@check_load_save('hpo_int_dict', 'HPO_INT_DICT_PKL', PKL_FILE_FORMAT)
	def get_hpo_int_dict(self):

		hpo_dict = self.get_hpo_dict()
		hpo_map_rank = self.get_hpo_map_rank()
		hpo_int_dict = {
			hpo_map_rank[hpo]: {
				'IS_A': item_list_to_rank_list(info_dict.get('IS_A', []), hpo_map_rank),
				'CHILD': item_list_to_rank_list(info_dict.get('CHILD', []), hpo_map_rank)
			} for hpo, info_dict in hpo_dict.items()
		}
		return hpo_int_dict


	@check_load_save('old_map_new_hpo', 'OLDMAPNEWHPO_JSON_PATH', JSON_FILE_FORMAT)
	def get_old_map_new_hpo_dict(self):
		"""
		Returns:
			dict: {OLD_HPO_CODE: NEW_HPO_CODE}
		"""
		def get_key_value(line):
			key, value = line.split(':', maxsplit=1)
			return key.strip(), value.strip()

		def map_add(old_code, new_code):
			if old_code not in old_map_new_hpo:
				old_map_new_hpo[old_code] = new_code
			elif new_code != old_map_new_hpo[old_code] and new_code in hpo_dict:
				if old_map_new_hpo[old_code] in hpo_dict:
					print('conflict: %s -> %s; %s -> %s. adopt %s' % (old_code, old_map_new_hpo[old_code], old_code, new_code, new_code))

				old_map_new_hpo[old_code] = new_code

		def valid_map(old_code):
			new_code = old_map_new_hpo[old_code]
			if new_code in hpo_dict:
				return new_code
			valid_new_code = valid_map(new_code)
			old_map_new_hpo[old_code] = valid_new_code
			return valid_new_code

		old_map_new_hpo = {}
		hpo_dict = self.get_hpo_dict()
		with open(self.HPO_OBO_PATH) as f:
			raw_list = f.read().split('[Term]')[1:]
			for raw_str in raw_list:
				lines = raw_str.strip().split('\n')
				id, alt_id, replace_by = None, None, None
				for line in lines:
					key, value = get_key_value(line)
					if key == 'id':
						id = value
					if key == 'alt_id': # alt_id -> id
						map_add(value, id)
					if key == 'replaced_by':    # id -> replaced_by
						map_add(id, value)
					if key == 'consider':
						map_add(id, value)

		for old_code in old_map_new_hpo:    # pass
			assert old_code not in hpo_dict
		for old_code, new_code in old_map_new_hpo.items():
			if new_code not in hpo_dict:
				valid_map(old_code)
		for new_code in old_map_new_hpo.values():
			assert new_code in hpo_dict
		return old_map_new_hpo


	def hpo_dict_exten(self, hpo_dict):
		for code in hpo_dict:
			for p_code in hpo_dict[code].get('IS_A', []):
				dict_list_add('CHILD', code, hpo_dict[p_code])


	def get_dis_to_hpo_dict(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Args:
			phe_list_mode (str):
		Returns:
			dict: {dis_code: [hpo_code1, ...]}
		"""
		if phe_list_mode == PHELIST_ORIGIN:
			return self._get_origin_dis_to_hpo_dict()
		if phe_list_mode == PHELIST_REDUCE:
			return self._get_reduce_dis_to_hpo_dict()
		if phe_list_mode == PHELIST_ANCESTOR: # reduce + ancestors
			return {dis_code: list(get_all_ancestors_for_many(hpo_list, self.get_hpo_dict())) for dis_code, hpo_list in self._get_origin_dis_to_hpo_dict().items()}
		if phe_list_mode == PHELIST_DESCENDENT:   # reduce + descendents
			return {dis_code: list(get_all_descendents_for_many(hpo_list, self.get_hpo_dict())) for dis_code, hpo_list in self._get_reduce_dis_to_hpo_dict().items()}
		if phe_list_mode == PHELIST_ANCESTOR_DUP:
			return {dis_code: get_all_dup_ancestors_for_many(hpo_list, self.get_hpo_dict()) for dis_code, hpo_list in self._get_origin_dis_to_hpo_dict().items()}
		assert False


	def get_hpo_to_dis_dict(self, phe_list_mode=PHELIST_ORIGIN):
		if phe_list_mode == PHELIST_ORIGIN:
			return self._get_origin_hpo_to_dis_dict()
		if phe_list_mode == PHELIST_REDUCE:
			return self._get_reduce_hpo_to_dis_dict()
		if phe_list_mode == PHELIST_ANCESTOR:
			return reverse_dict_list(self.get_dis_to_hpo_dict(PHELIST_ANCESTOR))
		assert False


	@check_load_save('anno_hpo_list', 'ANNO_HPO_JSON', JSON_FILE_FORMAT)
	def get_anno_hpo_list(self):
		anno_hpo_set = set()
		dis2hpo = self.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
		for hpo_list in dis2hpo.values():
			anno_hpo_set.update(hpo_list)
		return list(anno_hpo_set)


	def get_boqa_anno_tab_rows(self, default_prob):
		"""
		Returns:
			list: [row_info, ...], row_info = ['OMIM', '271320', ...]
		"""
		from core.reader import CCRDReader
		def get_date_and_assign(biocuration_by_str):
			match_obj = re.match('(.*:.*)\[(\d+-\d+-\d+)\]', biocuration_by_str)
			if not match_obj:
				print(biocuration_by_str)
			assert match_obj is not None
			return match_obj.group(1), match_obj.group(2)
		col_names, info_lists = self.read_phenotype_anno_hpoa()
		dis_to_hpo_prob_dict = self.get_dis_to_hpo_prob_dict(default_prob=default_prob)
		name2rank = {name:i for i, name in enumerate(col_names)}
		rows = []
		for info_list in info_lists:
			if info_list[name2rank['QUALIFIER']].strip() == 'NOT':
				continue
			dis_code = info_list[name2rank['DATABASE_ID']]
			hpo_code = info_list[name2rank['HPO_CODE']]
			freq = dis_to_hpo_prob_dict[dis_code][hpo_code]
			if freq is not None and freq <= 0:
				print(dis_code, hpo_code, freq)
			assert freq is None or freq > 0
			db, db_id = dis_code.split(':')
			date, assign_by = get_date_and_assign(info_list[name2rank['BIOCURATION_BY']])
			freq_modifier = str(freq*100) + '%' if freq is not None else ''
			rows.append([
				db, db_id, info_list[name2rank['DISEASE_NAME']], info_list[name2rank['QUALIFIER']].strip(), hpo_code,
				dis_code, info_list[name2rank['EVIDENCE']], info_list[name2rank['ONSET_MODIFIER']], freq_modifier, '',
				info_list[name2rank['ASPECT']], '', date, assign_by, freq_modifier
			])

		ccrd_dict = CCRDReader().get_ccrd_dict()
		for dis_code, hpo2prob in dis_to_hpo_prob_dict.items():
			if not dis_code.startswith('CCRD:'):
				continue
			db, db_id = dis_code.split(':')
			dis_name = ccrd_dict[dis_code]['ENG_NAME']
			for hpo, freq in hpo2prob.items():
				freq_modifier = str(freq * 100) + '%' if freq is not None else ''
				rows.append([
					db, db_id, dis_name, '', hpo,
					dis_code, 'IEA', '', freq_modifier, '',
					'', '', '', '', freq_modifier
				])
		col_names = [   # 2017
			'DB', 'DB_OBJECT_ID', 'DB_NAME', 'QUALIFIER', 'HPO_CODE',
			'DIS_CODE', 'EVIDENCE_CODE', 'ONSET_MODIFIER', 'FREQUENCY_MODIFIER', 'WITH',
			'ASPECT', 'SYNONYM', 'DATE', 'ASSIGNED_BY', 'FREQUENCY'
		]
		return rows, col_names


	def read_phenotype_anno_tab(self):
		def check():
			for line_info in info_lists:
				assert len(line_info) == len(col_names)
			pass

		col_names = [  # 2019
			'DB', 'DB_OBJECT_ID', 'DB_NAME', 'QUALIFIER', 'HPO_CODE',
			'DIS_CODE', 'EVIDENCE_CODE', 'ONSET_MODIFIER', 'FREQUENCY_MODIFIER', 'WITH',
			'ASPECT', 'SYNONYM', 'ASSIGNED_BY', 'FREQUENCY'
		]
		info_lists = read_standard_file(self.ANNOTATION_TAB_PATH)
		check()
		return col_names, info_lists


	def read_phenotype_anno_hpoa(self):
		def check():
			for line_info in info_lists:
				assert len(line_info) == len(col_names)
			pass
		col_names = [
			'DATABASE_ID', 'DISEASE_NAME', 'QUALIFIER', 'HPO_CODE', 'DB_REF', 'EVIDENCE',
			'ONSET_MODIFIER', 'FREQUENCY_MODIFIER', 'SEX', 'MODIFIER', 'ASPECT', 'BIOCURATION_BY'
		]
		info_lists = read_standard_file(self.ANNOTATION_HPOA_PATH)[1:]
		check()
		return col_names, info_lists


	def hpo2freq(self, hpo_code):
		if self.hpo2freq_dict is None:
			self.hpo2freq_dict = {
			'HP:0040285': [0.0, 0.0, 0.0],      # Excluded; 0%
			'HP:0040284': [0.01, 0.025, 0.04],     # Very rare; 1%-4%
			'HP:0040283': [0.05, 0.17, 0.29],     # Occasional; 5%-29%
			'HP:0040282': [0.3, 0.545, 0.79],      # Frequent; 30%-79%
			'HP:0040281': [0.8, 0.895, 0.99],      # Very frequent; 80%-99%
			'HP:0040280': [1.0, 1.0, 1.0]       # Obligate; 100%
		}
		return self.hpo2freq_dict[hpo_code]


	def word2freq(self, word):
		if self.word2freq_dict is None:
			self.word2freq_dict = {
			'very rare': 0.01, 'rare': 0.05, 'occasional': 0.075, 'main': 0.25,
			'frequent': 0.33, 'typical': 0.5, 'common': 0.75, 'hallmark': 0.9,  # common: 0.5 -> 0.75 (same as BOQA)
			'obligate': 1.0, 'variable': 0.545, 'very frequent': 0.895  #
		}
		word = word.lower()
		if word in self.word2freq_dict:
			return self.word2freq_dict[word]
		match_obj = re.match('^(\d+)/(\d+)$', word)
		if match_obj:
			return int(match_obj.group(1)) / int(match_obj.group(2))
		match_obj = re.match('^([\d\.]+?)\s*%$', word)
		if match_obj:
			return float(match_obj.group(1)) / 100
		match_obj = re.match('^(\d+)\s*of\s*(\d+)$', word)
		if match_obj:
			return int(match_obj.group(1)) / int(match_obj.group(2))
		match_obj = re.match('^(\d+)%?-(\d+)%$', word)
		if match_obj:
			return (int(match_obj.group(1)) + int(match_obj.group(2))) / 200
		print('Error:', word); assert False


	def get_dis_int_to_hpo_int_prob(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {dis_int: [[hpo_int, prob], ...]}
		"""
		dis_to_hpo_prob = self.get_dis_to_hpo_prob(mode, default_prob, phe_list_mode)
		dis_map_rank, hpo_map_rank = self.get_dis_map_rank(), self.get_hpo_map_rank()
		ret_dict = {}
		for dis_code, hpo_prob_list in dis_to_hpo_prob.items():
			ret_dict[dis_map_rank[dis_code]] = [[hpo_map_rank[hpo_code], prob] for hpo_code, prob in hpo_prob_list]
		return ret_dict


	def convert_dis_to_hpoprob(self, dis_to_hpoprob, phe_list_mode):
		if phe_list_mode == PHELIST_ORIGIN:
			return dis_to_hpoprob
		elif phe_list_mode == PHELIST_REDUCE:
			dis2hpo = self.get_dis_to_hpo_dict(PHELIST_REDUCE)
			for dis_code, hpo_prob_list in dis_to_hpoprob.items():
				hpo_set = set(dis2hpo[dis_code])
				dis_to_hpoprob[dis_code] = [[hpo, prob] for hpo, prob in hpo_prob_list if hpo in hpo_set]
			return dis_to_hpoprob
		else:
			assert False


	def get_dis_to_hpo_raw_prob(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {dis_code: [[hpo_code, prob], ...]}; prob = 'HP:xxx' or float or None
		"""
		if os.path.exists(self.DIS_TO_HPO_PROB_RAW_JSON):
			dis_to_hpoprob = json.load(open(self.DIS_TO_HPO_PROB_RAW_JSON))
		else:
			dis_to_hpoprob = self._get_dis_to_hpo_prob_from_hpoa(use_freq_raw=True)
			json.dump(dis_to_hpoprob, open(self.DIS_TO_HPO_PROB_RAW_JSON, 'w'), indent=2)
		ccrd_dis_to_hpo = self._get_origin_ccrd_dis_to_hpo()
		dis_to_hpoprob.update({ccrd_code:[[hpo, None] for hpo in hpo_list] for ccrd_code, hpo_list in ccrd_dis_to_hpo.items()})
		return self.convert_dis_to_hpoprob(dis_to_hpoprob, phe_list_mode)


	def get_dis_to_hpo_prob(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):

		dis_to_hpo_prob = self._get_dis_to_hpo_prob(mode, default_prob)
		ccrd_dis_to_hpo = self._get_origin_ccrd_dis_to_hpo()
		dis_to_hpo_prob.update({ccrd_code: [[hpo, default_prob] for hpo in hpo_list] for ccrd_code, hpo_list in ccrd_dis_to_hpo.items()})
		return self.convert_dis_to_hpoprob(dis_to_hpo_prob, phe_list_mode)


	def _get_dis_to_hpo_prob_from_tab(self):
		col_names, info_lists = self.read_phenotype_anno_tab()
		COL_NUM = len(col_names)
		name2rank = {name:i for i, name in enumerate(col_names)}
		dis_to_hpo_prob = {}
		counter = Counter()
		for info_list in tqdm(info_lists):
			if info_list[name2rank['QUALIFIER']].strip() == 'NOT':
				continue
			info_list = list_add_tail(info_list, '', COL_NUM - len(info_list))
			dis_code = info_list[name2rank['DB']] + ':' + info_list[name2rank['DB_OBJECT_ID']]
			hpo_code = info_list[name2rank['HPO_CODE']]
			freq_hpo_code = info_list[name2rank['FREQUENCY_MODIFIER']].strip()
			freq_word = info_list[name2rank['FREQUENCY']]
			if freq_word:
				if freq_word.startswith('HP:'):
					if freq_word != freq_hpo_code:
						print(info_list)
					assert freq_word == freq_hpo_code
					freq_float = self.hpo2freq(freq_word)
					counter['use freq hpo code'] += 1
				else:
					freq_float = self.word2freq(freq_word)
					counter['use freq word'] += 1  # 3822
			elif freq_hpo_code:
				freq_float = self.hpo2freq(freq_hpo_code)
				counter['use freq hpo code'] += 1  # 86128
			else:
				freq_float = None
				counter['use default prob'] += 1  # 92072
			dict_list_add(dis_code, [hpo_code, freq_float], dis_to_hpo_prob)
		print(counter)
		return dis_to_hpo_prob


	def get_dis_to_name(self):
		col_names, info_lists = self.read_phenotype_anno_hpoa()
		COL_NUM = len(col_names)
		name2rank = {name:i for i, name in enumerate(col_names)}
		dis2name = {}
		for info_list in tqdm(info_lists):
			assert len(info_list) == COL_NUM
			dis_code = info_list[name2rank['DATABASE_ID']].strip()
			dis_name = info_list[name2rank['DISEASE_NAME']].strip()
			if dis_code not in dis2name:
				dis2name[dis_code] = dis_name
		return dis2name


	def _get_dis_to_hpo_prob_from_hpoa(self, use_freq_raw=False):
		col_names, info_lists = self.read_phenotype_anno_hpoa()
		COL_NUM = len(col_names)
		name2rank = {name:i for i, name in enumerate(col_names)}
		dis_to_hpo_prob = {}
		counter = Counter()
		for info_list in tqdm(info_lists):
			assert len(info_list) == COL_NUM
			if info_list[name2rank['QUALIFIER']].strip() == 'NOT':
				continue
			# info_list = list_add_tail(info_list, '', COL_NUM - len(info_list))
			dis_code = info_list[name2rank['DATABASE_ID']]
			hpo_code = info_list[name2rank['HPO_CODE']]
			freq_modifier = info_list[name2rank['FREQUENCY_MODIFIER']].strip()
			if freq_modifier.startswith('HP:'):
				freq_raw = freq_modifier
				freq_float = self.hpo2freq(freq_modifier)
				counter['use freq hpo code'] += 1   # 92215
			elif freq_modifier:
				freq_raw = freq_float = self.word2freq(freq_modifier)
				counter['use freq word'] += 1   # 4042
			else:
				freq_raw = freq_float = None
				counter['use default prob'] += 1    # 98950
			dict_list_add(dis_code, [hpo_code, freq_raw if use_freq_raw else freq_float], dis_to_hpo_prob)
		print(counter)
		return dis_to_hpo_prob


	def _get_dis_to_hpo_prob(self, mode=1, default_prob=1.0, source_type='hpoa'):

		def revise_dis_to_hpo_prob(dis_to_hpo_prob):
			for dis_code in dis_to_hpo_prob:
				hpo_prob_list = dis_to_hpo_prob[dis_code]
				for i in range(len(hpo_prob_list)):
					if hpo_prob_list[i][1] is None:
						hpo_prob_list[i][1] = default_prob # None -> default_prob
					if isinstance(hpo_prob_list[i][1], list):
						hpo_prob_list[i][1] = hpo_prob_list[i][1][mode] # [p1, p2, p3] -> px
		def check(dis_to_hpo_prob):
			for dis_code, hpo_prob_list in dis_to_hpo_prob.items():
				for hpo_code, prob in hpo_prob_list:
					if not isinstance(prob, list):
						prob = [prob]
					for p in prob:
						if not (p is None or (p >= 0 and p <= 1)):
							print(dis_code, hpo_code, prob)
						assert p is None or (p >= 0 and p <= 1)  # OMIM:216550 HP:0000662 9/7 -> 7/9; OMIM:216550 HP:0000662 18/2 -> 2/18; OMIM:613706 HP:0001639 0/5 -> 1/5; OMIM:613706 HP:0000286 0/5 -> 1/5
		file_json = self.DIS_TO_HPO_PROB_HPOA_JSON if source_type == 'hpoa' else self.DIS_TO_HPO_PROB_TAB_JSON
		if os.path.exists(file_json):
			dis_to_hpo_prob = json.load(open(file_json))
			revise_dis_to_hpo_prob(dis_to_hpo_prob)
			return dis_to_hpo_prob
		if source_type == 'tab':
			dis_to_hpo_prob = self._get_dis_to_hpo_prob_from_tab()
		else:
			assert source_type == 'hpoa'
			dis_to_hpo_prob = self._get_dis_to_hpo_prob_from_hpoa()
		check(dis_to_hpo_prob)
		json.dump(dis_to_hpo_prob, open(file_json, 'w'), indent=2)
		revise_dis_to_hpo_prob(dis_to_hpo_prob)
		return dis_to_hpo_prob


	def get_dis_to_hpo_prob_dict(self, mode=1, default_prob=1.0):
		"""
		Returns:
			dict: {dis_code: {hpo_code: prob}}
		"""
		dis_to_hpo_prob = self.get_dis_to_hpo_prob(mode, default_prob)
		return {dis_code: {hpo: prob for hpo, prob in hpo_prob_list} for dis_code, hpo_prob_list in dis_to_hpo_prob.items()}


	def _get_origin_dis_to_hpo_dict(self, source_type='hpoa'):
		if source_type == 'hpoa':
			ret_dict =  self._get_origin_dis_to_hpo_dict_from_hpoa()
		else:
			assert source_type == 'tab'
			ret_dict =  self._get_origin_dis_to_hpo_dict_from_tab()
		ret_dict.update(self._get_origin_ccrd_dis_to_hpo())
		return ret_dict


	def _get_origin_ccrd_dis_to_hpo(self):
		dis2info = json.load(open(self.CCRD_JSON_PATH))
		return {dis_code: info['PHENOTYPE_LIST'] for dis_code, info in dis2info.items()}


	def get_ccrd_dis_list(self):
		return sorted(list(self._get_origin_ccrd_dis_to_hpo().keys()))


	@check_load_save('dis2hpo', 'DISTOHPO_HPOA_JSON_PATH', JSON_FILE_FORMAT)
	def _get_origin_dis_to_hpo_dict_from_hpoa(self):
		"""
		Returns:
			dict: {dis_code: [hpo_code, ...]}
		"""
		col_names, info_lists = self.read_phenotype_anno_hpoa()
		COL_NUM = len(col_names)
		dis2hpo = {}
		name2rank = {name:i for i, name in enumerate(col_names)}
		for info_list in info_lists:
			assert len(info_list) == COL_NUM
			if info_list[name2rank['QUALIFIER']].strip() == 'NOT':
				continue

			dis_code = info_list[name2rank['DATABASE_ID']]
			dict_list_add(dis_code, info_list[name2rank['HPO_CODE']], dis2hpo)
		return dis2hpo
	

	@check_load_save('dis2hpo', 'DISTOHPO_TAB_JSON_PATH', JSON_FILE_FORMAT)
	def _get_origin_dis_to_hpo_dict_from_tab(self):
		"""
		Returns:
			dict: {dis_code: [hpo_code, ...]}
		"""
		col_names, info_lists = self.read_phenotype_anno_tab(); COL_NUM = len(col_names)
		dis2hpo = {}
		name2rank = {name: i for i, name in enumerate(col_names)}
		print(info_lists[0])
		for info_list in info_lists:
			assert len(info_list) == len(col_names)
			if info_list[name2rank['QUALIFIER']].strip() == 'NOT':
				continue
			# info_list = list_add_tail(info_list, '', COL_NUM-len(info_list))
			assert len(info_list) == 14 # 15 for 2017 version
			dis_code = info_list[name2rank['DB']] + ':' + info_list[name2rank['DB_OBJECT_ID']]
			dict_list_add(dis_code, info_list[name2rank['HPO_CODE']], dis2hpo)
		return dis2hpo


	@check_load_save('hpo2dis', 'HPOTODIS_JSON_PATH', JSON_FILE_FORMAT)
	def _get_origin_hpo_to_dis_dict(self):
		return reverse_dict_list(self._get_origin_dis_to_hpo_dict())


	@check_load_save('dis2hpo_reduce', 'DISTOHPO_REDUCE_JSON_PATH', JSON_FILE_FORMAT)
	def _get_reduce_dis_to_hpo_dict(self):

		hpo_dict = self.get_hpo_dict()
		dis2hpo = self._get_origin_dis_to_hpo_dict()
		new_dis_to_hpo = {}
		for dis, hpo_list in tqdm(dis2hpo.items()):
			new_dis_to_hpo[dis] = delete_redundacy(hpo_list, hpo_dict)
		return new_dis_to_hpo


	@check_load_save('hpo2dis_reduce', 'HPOTODIS_REDUCE_JSON_PATH', JSON_FILE_FORMAT)
	def _get_reduce_hpo_to_dis_dict(self):

		return reverse_dict_list(self._get_reduce_dis_to_hpo_dict())


	@check_load('dis_list', 'DIS_LIST_JSON', JSON_FILE_FORMAT)
	def get_dis_list(self):
		"""
		Returns:
			list: [disease_code1, disease_code2, ...]
		"""
		dis_list, _ = self.gen_rank_json(list(self.get_dis_to_hpo_dict().keys()), self.DIS_LIST_JSON, self.DIS_MAP_RANK_JSON)
		return dis_list


	@check_load('dis_map_rank', 'DIS_MAP_RANK_JSON', JSON_FILE_FORMAT)
	def get_dis_map_rank(self):
		"""
		Returns:
			dict: {disease_code: disease_rank}
		"""
		_, dis_map_rank = self.gen_rank_json(list(self.get_dis_to_hpo_dict().keys()), self.DIS_LIST_JSON, self.DIS_MAP_RANK_JSON)


		return dis_map_rank


	def get_dis_int_to_hpo_int(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {disease_rank: [hpo_rank, ...]}
		"""
		dis2hpo = self.get_dis_to_hpo_dict(phe_list_mode)
		return dict_change_key_value(dis2hpo, self.get_dis_map_rank(), self.get_hpo_map_rank())


	def get_hpo_int_to_dis_int(self, phe_list_mode=PHELIST_ORIGIN):
		"""Note: There may be some HPO not in returned dict
		Returns:
			dict: {hpo_rank: [disease_rank, ...]}
		"""
		hpo2dis = self.get_hpo_to_dis_dict(phe_list_mode)
		return dict_change_key_value(hpo2dis, self.get_hpo_map_rank(), self.get_dis_map_rank())


	def get_gene_to_hpo_dict(self, phe_list_mode=PHELIST_ORIGIN):
		if phe_list_mode == PHELIST_ORIGIN:
			return self.get_origin_gene_to_hpo_dict()
		if phe_list_mode == PHELIST_REDUCE:
			return self.get_reduce_gene_to_hpo_dict()
		hpo_dict = self.get_hpo_dict()
		return {gene_code: list(get_all_ancestors_for_many(hpo_list, hpo_dict)) for gene_code, hpo_list in self.get_origin_gene_to_hpo_dict().items()}


	def get_hpo_to_gene_dict(self, phe_list_mode=PHELIST_ORIGIN):
		if phe_list_mode == PHELIST_ORIGIN:
			self.get_origin_hpo_to_gene_dict()
		if phe_list_mode == PHELIST_REDUCE:
			return self.get_reduce_hpo_to_gene_dict()
		return reverse_dict_list(self.get_gene_to_hpo_dict(PHELIST_ANCESTOR))


	def read_gene_to_phenotype_txt(self):
		"""
		Returns:
			list: [colName1, colName2, ...]
			list: [row_info, ...]; row_info = [str, str, ...];
		"""
		col_names = ['ENTREZ_GENE_ID', 'ENTREZ_GENE_SYMBOL', 'HPO_TERM_NAME', 'HPO_TERM_ID']
		info_lists = read_standard_file(self.GENE_TO_PHENOTYPE_TXT)
		return col_names, info_lists


	def gen_gene_code(self, gene_id):
		return 'EZ:'+gene_id


	@check_load_save('gene2hpo', 'GENETOHPO_JSON', JSON_FILE_FORMAT)
	def get_origin_gene_to_hpo_dict(self):
		"""
		Returns:
			dict: {gene_code: [hpo_code, ...]}
		"""
		col_names, info_lists = self.read_gene_to_phenotype_txt()
		col2rank = {col_names[i]: i for i in range(len(col_names))}
		gene2hpo = {}
		for info_list in info_lists:
			assert len(info_list) == 4
			gene_code = self.gen_gene_code(info_list[ col2rank['ENTREZ_GENE_ID'] ])
			dict_list_add(gene_code, info_list[ col2rank['HPO_TERM_ID'] ], gene2hpo)
		return gene2hpo


	@check_load_save('hpo2gene', 'HPOTOGENE_JSON', JSON_FILE_FORMAT)
	def get_origin_hpo_to_gene_dict(self):
		"""
		Returns:
			dict: {hpo_code: [gene_code, ...]}
		"""
		return reverse_dict_list(self.get_origin_gene_to_hpo_dict())


	@check_load_save('gene2hpo_reduce', 'GENETOHPO_REDUCE_JSON', JSON_FILE_FORMAT)
	def get_reduce_gene_to_hpo_dict(self):
		hpo_dict = self.get_hpo_dict()
		gene2hpo = self.get_origin_gene_to_hpo_dict()
		new_gene_to_hpo = {gene_code: delete_redundacy(hpo_list, hpo_dict) for gene_code, hpo_list in tqdm(gene2hpo.items())}
		return new_gene_to_hpo


	@check_load_save('hpo2gene_reduce', 'HPOTOGENE_REDUCE_JSON', JSON_FILE_FORMAT)
	def get_reduce_hpo_to_gene_dict(self):
		return reverse_dict_list(self.get_reduce_gene_to_hpo_dict())


	@check_load('gene_list', 'GENE_LIST_JSON', JSON_FILE_FORMAT)
	def get_gene_list(self):
		"""
		Returns:
			list: [geneCode1, geneCode2, ...]
		"""
		gene_list, _ = self.gen_rank_json(list(self.get_gene_to_hpo_dict().keys()), self.GENE_LIST_JSON, self.GENE_MAP_RANK_JSON)
		return gene_list


	@check_load('gene_map_rank', 'GENE_MAP_RANK_JSON', JSON_FILE_FORMAT)
	def get_gene_map_rank(self):
		"""
		Returns:
			dict: {gene: geneRank}
		"""
		_, gene_map_rank = self.gen_rank_json(list(self.get_gene_to_hpo_dict().keys()), self.GENE_LIST_JSON, self.GENE_MAP_RANK_JSON)
		return gene_map_rank


	def get_gene_int_to_hpo_int(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {geneRank: [hpo_rank, ...]}
		"""
		gene2hpo = self.get_gene_to_hpo_dict(phe_list_mode)
		return dict_change_key_value(gene2hpo, self.get_gene_map_rank(), self.get_hpo_map_rank())


	def get_hpo_int_to_gene_int(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {hpo_rank: [geneRank, ...]}
		"""
		hpo2gene = self.get_hpo_to_gene_dict(phe_list_mode)
		return dict_change_key_value(hpo2gene, self.get_hpo_map_rank(), self.get_gene_map_rank())


	@check_load_save('gene2dis', 'GENE_TO_DIS_JSON', JSON_FILE_FORMAT)
	def get_gene_to_dis_list(self):
		"""
		Returns:
			dict: {gene_code: [dis1, dis2, ...]}
		"""
		gene2dis = {}
		gene_to_dis_info = read_standard_file(self.HPO_GENES_TO_DISEASES_TXT)    # [[entrezId, geneSymbol, DisId], ...]
		for i in range(len(gene_to_dis_info)):
			dict_list_add(self.gen_gene_code(gene_to_dis_info[i][0]), gene_to_dis_info[i][2], gene2dis)
		return gene2dis


	def get_dis_to_gene_list(self):
		"""
		Returns:
			dict: {dis_code: [gene_code, ...]}; e.g. 'ORPHA:2636': ['EZ:100151683']
		"""
		return reverse_dict_list(self.get_gene_to_dis_list())


	def get_dis_to_gene_symbols(self):
		"""
		Returns:
			dict: {dis_code: [gene_symbol, ...]}
		"""
		gene2symbol = self.get_gene_to_symbol()
		dis2gene = self.get_dis_to_gene_list()
		return {dis: [gene2symbol[g] for g in genes] for dis, genes in dis2gene.items()}


	def get_gene_symbol_to_dis_list(self):
		"""
		Returns:
			dict: {geneSymbol: [disCode1, disCode2, ...]}
		"""
		gene_to_dis_list = self.get_gene_to_dis_list()
		gene2symbol = self.get_gene_to_symbol()
		return {gene2symbol[gene]: dis_list for gene, dis_list in gene_to_dis_list.items()}


	@check_load_save('gene2symbol', 'GENE_TO_SYMBOL_JSON', JSON_FILE_FORMAT)
	def get_gene_to_symbol(self):
		"""
		Returns:
			dict: {gene_code: geneSymbol}
		"""
		gene2symbol = {}
		gene_to_dis_info = read_standard_file(self.HPO_GENES_TO_DISEASES_TXT)  # [[entrezId, geneSymbol, DisId], ...]
		for i in range(len(gene_to_dis_info)):
			gene2symbol[self.gen_gene_code(gene_to_dis_info[i][0])] = gene_to_dis_info[i][1]
		return gene2symbol


	def get_hpo_adj_mat_base(self, key_list=None, dtype=np.int32):
		"""
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num), dtype=np.int32
		"""
		key_list = key_list or ['CHILD', 'IS_A']
		hpo_num = self.get_hpo_num()
		hpo_int_dict = self.get_hpo_int_dict()
		data = []
		for hpo_int in range(hpo_num):
			info_dict = hpo_int_dict[hpo_int]
			tmp = []
			for key in key_list:
				tmp.extend(info_dict.get(key, []))
			data.append(tmp)
		return data_to_01_matrix(data, hpo_num, dtype)


	@check_load_save('adj_mat', 'ADJ_MAT_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_hpo_adj_mat(self):
		"""
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num), dtype=np.int32
		"""
		return self.get_hpo_adj_mat_base(['CHILD', 'IS_A'], np.int32)


	@check_load_save('parent_mat', 'PARENT_MAT_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_hpo_parent_mat(self):
		"""
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num), dtype=np.int32
		"""
		return self.get_hpo_adj_mat_base(['IS_A'], np.int32)


	def get_hpo_ances_mat(self, contain_self=True, dtype=np.int32):
		"""
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num), dtype=np.int32
		"""
		hpo_int_dict = self.get_hpo_int_dict()
		hpo_num = self.get_hpo_num()
		data = [list(get_all_ancestors(hpo_int, hpo_int_dict, contain_self)) for hpo_int in range(hpo_num)]
		return data_to_01_matrix(data, hpo_num, dtype=dtype)


	def get_hpo_degree(self):
		"""
		Returns:
			np.ndarray: shape=(hpo_num,)
		"""
		return self.get_hpo_adj_mat().sum(axis=1).A.flatten()


	def get_no_anno_hpo_int_set(self):
		return set(range(self.get_hpo_num())) - set(self.get_hpo_int_to_dis_int(PHELIST_ANCESTOR).keys())


	def get_anno_hpo_int_set(self):
		return set(self.get_hpo_int_to_dis_int(PHELIST_ANCESTOR).keys())


	@check_return('hpo2depth')
	def get_hpo2depth(self, root='HP:0000001'):
		"""
		Returns:
			dict: {hpo_code: depth}
		"""
		return get_all_descendents_with_dist(root, self.get_slice_hpo_dict())  # {hpo_code: depth}


	def get_hpo_co_mat(self, phe_list_mode=PHELIST_REDUCE, dtype=np.int64):
		"""co-occurrence
		"""
		dis_int_to_hpo_int = self.get_dis_int_to_hpo_int(phe_list_mode)
		data = [dis_int_to_hpo_int[i] for i in range(self.get_dis_num())]
		return data_to_cooccur_matrix(data, self.get_hpo_num(), dtype=dtype)


	def statistic(self):
		print('HPO Number:', self.get_hpo_num())
		print('Disease Number:', self.get_dis_num())
		print('HPO used:', len(set([hpo for dis_code, hpo_list in self.get_dis_to_hpo_dict(PHELIST_ANCESTOR).items() for hpo in hpo_list])))


if __name__ == '__main__':
	from core.explainer import Explainer
	reader = HPOReader()


	reader.get_chpo_dict()
