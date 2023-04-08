import re
import os
from bs4 import BeautifulSoup
import rdflib as rdf
from rdflib import RDFS
import re
from tqdm import tqdm

from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL
from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT, ORPHA_CURINF_ASSERT, ORPHA_MANUAL_ASSERT
from core.utils.utils import check_load_save, set_if_not_empty, dict_list_add, unique_list, reverse_dict_list, slice_dict_with_keep_set


class OrphanetReader(object):
	def __init__(self):
		self.link2reversed = {'NTBT': 'BTNT', 'BTNT': 'NTBT', 'NTBT/E': 'BTNT/E', 'BTNT/E': 'NTBT/E'}
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'ORPHANET')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)

		self.CNS_ORPHANET_HTML_PATH = os.path.join(DATA_PATH, 'raw', 'ORPHANET', 'ORPHANET_CHPO.html')
		self.CNS_ORPHANET_JSON_PATH = os.path.join(self.PREPROCESS_FOLDER, 'cns_orphanet.json')
		self.cns_orpha = None   # {code: {'CNS_NAME': '', 'ENG_NAME': ''}}

		self.ORDO_ORPHANET_OWL = os.path.join(DATA_PATH, 'raw', 'ORPHANET', '2019', 'ordo_orphanet.owl')
		self.ORPHA_DICT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'orpha_dict.json')
		self.orpha_dict = None

		self.ORPHA_SLICE_DICT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'orpha_slice_dict.json')
		self.orpha_slice_dict = None

		self.ORPHA_TO_OMIM = os.path.join(self.PREPROCESS_FOLDER, 'orpha2omim.json')
		self.orpha2omim = None

		self.OMIM_TO_ORPHA = os.path.join(self.PREPROCESS_FOLDER, 'omim2orpha.json')
		self.omim2orpha = None

		self.OBSOLETE_CODES_JSON = os.path.join(self.PREPROCESS_FOLDER, 'obsolete_codes.json')
		self.obsolete_codes = None

		self.level2types = {
			DISORDER_GROUP_LEVEL: [
				'ORPHA:377794',  # group of disorders
			],
			DISORDER_LEVEL: [
				'ORPHA:377790',  # biological anomaly
				'ORPHA:377792',  # clinical syndrome
				'ORPHA:377788',  # disease
				'ORPHA:377789',  # malformation syndrome
				'ORPHA:377791',  # morphological anomaly
				'ORPHA:377793',  # particular clinical situation in a disease or syndrome
			],
			DISORDER_SUBTYPE_LEVEL: [
				'ORPHA:377796',  # clinical subtype
				'ORPHA:377795',  # etiological subtype
				'ORPHA:377797',  # histopathological subtype
			]
		}
		self.type2level = {t:level for level, types in self.level2types.items() for t in types}


	def add_orpha_prefix(self, orphaNum):
		return 'ORPHA:'+orphaNum


	@check_load_save('cns_orpha', 'CNS_ORPHANET_JSON_PATH', JSON_FILE_FORMAT)
	def get_cns_orpha_dict(self):
		"""
		Returns:
			dict: {orpha_code: {'ENG_NAME': '', 'CNS_NAME': ''}}
		"""
		cns_orpha = {}
		soup = BeautifulSoup(open(self.CNS_ORPHANET_HTML_PATH), 'lxml')
		table = soup.find_all('table')[0]
		for tr in table.find_all('tr')[1:]:
			tds = tr.find_all('td')
			code = self.add_orpha_prefix(tds[0].string.strip())
			cns_orpha[code] = {}
			set_if_not_empty(cns_orpha[code], 'CNS_NAME', (tds[2].string or '').strip())
			set_if_not_empty(cns_orpha[code], 'ENG_NAME', (tds[1].string or '').strip())
		return cns_orpha


	def uri_str_to_orpha_code(self, rdf_str):
		return 'ORPHA:' + rdf_str.split('_').pop()


	def get_link_type_from_str(self, link_str):
		return re.match('^(.*?) \(.*$', link_str).group(1).strip()


	def get_all_orpha_code_uri_str(self, g):
		return unique_list([s.toPython() for s in g.subjects() if isinstance(s, rdf.URIRef)])


	def correct_orpha_dict(self, orpha_dict):
		orpha_to_omim_cor = {
			'ORPHA:51608': [('OMIM:208000', 'BTNT'), ('OMIM:614473', 'BTNT')],
			'ORPHA:85179': [('OMIM:259720', 'NTBT')],
			'ORPHA:93347': [('OMIM:607095', 'BTNT'), ('OMIM:617396', 'BTNT')],
			'ORPHA:529970': [('OMIM:617187', 'BTNT'), ('OMIM:618112', 'BTNT')],
			'ORPHA:1114': [('OMIM:600360', 'BTNT')],
			'ORPHA:1272': [('OMIM:601353', 'BTNT'), ('OMIM:601088', 'BTNT')],
			'ORPHA:2701': [('OMIM:617506', 'BTNT'), ('OMIM:607721', 'BTNT')],
			'ORPHA:521438': [('OMIM:617661', 'BTNT'), ('OMIM:617660', 'BTNT')],
			'ORPHA:2052': [('OMIM:617667', 'BTNT'), ('OMIM:219000', 'BTNT'), ('OMIM:617666', 'BTNT')],
			'ORPHA:404466': [('OMIM:615774', 'BTNT'), ('OMIM:618353', 'BTNT'), ('OMIM:617712', 'BTNT')],
			'ORPHA:169142': [('OMIM:245480', 'BTNT'), ('OMIM:617475', 'BTNT')],
			'ORPHA:97229': [('OMIM:614707', 'BTNT'), ('OMIM:211530', 'BTNT'), ('OMIM:211500', 'BTNT')],
			'ORPHA:526': [('OMIM:618114', 'BTNT'), ('OMIM:618126', 'BTNT'), ('OMIM:177200', 'BTNT')],
			'ORPHA:71212':[('OMIM:231530', 'NTBT')],
			'ORPHA:857': [('OMIM:617466', 'BTNT'), ('OMIM:107480', 'BTNT')],
			'ORPHA:86820': [('OMIM:608805', 'BTNT'), ('OMIM:617383', 'BTNT')],
			'ORPHA:329242': [('OMIM:615863', 'BTNT'), ('OMIM:618183', 'BTNT')],
			'ORPHA:263548': [('OMIM:618084', 'BTNT'), ('OMIM:616265', 'BTNT')],
			'ORPHA:431166': [('OMIM:616636', 'BTNT'), ('OMIM:616669', 'BTNT')],
			'ORPHA:803': [('OMIM:105400', 'BTNT')], # CCRD:4

		}
		orpha_to_omim_del = {
			# 'ORPHA:2662': ['OMIM:255980'], # MOVED TO 301026
			'ORPHA:357074': ['OMIM:617402', 'OMIM:617403'],
			'ORPHA:2053': ['OMIM:618436'],
			# 'ORPHA:83629': ['OMIM:300660'], # MOVED TO 300232
			'ORPHA:231662': ['OMIM:618160'],
			# 'ORPHA:85328': ['OMIM:300706'], # MOVED TO 309590
			'ORPHA:267': ['OMIM:253600'],
			'ORPHA:287': ['OMIM:130010'],
		}
		for orpha, omim_link in orpha_to_omim_cor.items():
			xref_list = orpha_dict[orpha]['DB_XREFS']
			cor_omim_to_link = {omim: link for omim, link in omim_link}
			for xref_item in xref_list:
				if xref_item[0] in cor_omim_to_link:
					xref_item[1] = cor_omim_to_link[xref_item[0]]
		for orpha, omim_list in orpha_to_omim_del.items():
			orpha_dict[orpha]['DB_XREFS'] = [xref_item for xref_item in orpha_dict[orpha]['DB_XREFS'] if xref_item[0] not in omim_list]
		return orpha_dict


	@check_load_save('orpha_dict', 'ORPHA_DICT_JSON', JSON_FILE_FORMAT)
	def get_orpha_dict(self):
		"""
		Returns:
			dict: {
				orpha_code: {
					'DB_XREFS': [(xRrefCode, link_type, assert_type), ...],
					'ALT_TERMS': [],
					'ENG_NAME': '',
					'ENG_DEF': ''
					'IS_A': [],
					'PART_OF': [],
					'HAS_PART': [],
				}
			}
		"""
		print(f'parsing owl: {self.ORDO_ORPHANET_OWL}', end='...')
		g = rdf.Graph()
		g.parse(self.ORDO_ORPHANET_OWL)
		print('done')
		self.before_read_orpha_owl(g)

		orpha_dict = {}
		for uri_str in tqdm(self.get_all_orpha_code_uri_str(g)):
			orpha_code = self.uri_str_to_orpha_code(uri_str)
			info_dict = self.uri_to_info_dict(g, uri_str)
			orpha_dict[orpha_code] = info_dict
		for orpha, info in orpha_dict.items():
			for partof_orpha in info.get('PART_OF', []):
				dict_list_add('HAS_PART', orpha, orpha_dict[partof_orpha])
			for isa_orpha in info.get('IS_A', []):
				dict_list_add('CHILD', orpha, orpha_dict[isa_orpha])

		xref_look_up = {
			(orpha_code, xref_code): [link_type, assert_type]
			for orpha_code, xref_info in self.get_db_xref_types(g).items()
			for xref_code, link_type, assert_type in xref_info
		}
		for orpha, info in orpha_dict.items():
			info['DB_XREFS'] = [[xref_code] + xref_look_up.get((orpha, xref_code), [None, None]) for xref_code in info['DB_XREFS']]
		orpha_dict = {orpha_code: info for orpha_code, info in orpha_dict.items() if re.match('ORPHA:[0-9]+', orpha_code)}
		return self.correct_orpha_dict(orpha_dict)


	@check_load_save('orpha_slice_dict', 'ORPHA_SLICE_DICT_JSON', JSON_FILE_FORMAT)
	def get_orpha_slice_dict(self):
		orpha_dict = self.get_orpha_dict()
		return {code:slice_dict_with_keep_set(orpha_dict[code], {'IS_A', 'CHILD', 'PART_OF', 'HAS_PART'}) for code in orpha_dict}


	def uri_to_info_dict(self, g, uri_str):
		"""
		Args:
			uri_str (str)
		Returns:
			dict: {'DB_XREFS': [], 'ALT_TERMS': [], 'ENG_NAME': '', 'ENG_DEF': '', 'DB_XREFS_TYPE': [], 'IS_A': [], 'PART_OF': []}
		"""
		res = g.resource(rdf.URIRef(uri_str))
		ret_dict = {
			'DB_XREFS': [o.toPython() for o in res.objects(self.has_db_xref_p)],
			'ALT_TERMS': [o.toPython() for o in res.objects(self.alt_term_p)],
		}
		labels = [o.toPython() for o in res.objects(self.rdfs_label_p)]   # assert len(labels) <= 1   # Orphanet_C040: 2 label
		ret_dict['ENG_NAME'] = labels[0] if labels else ''
		defs = [o.toPython() for o in res.objects(self.efo_def_p)]; assert len(defs) <= 1
		ret_dict['ENG_DEF'] = defs[0] if defs else ''
		ret_dict.update(self.get_parent_info(res, uri_str))
		return ret_dict


	def get_db_xref_types(self, g):
		"""
		Returns:
			dict: {orpha_code: [(xRrefCode, link_type, assert_type), ...]}
		"""
		target_ref_p = self.owl['annotatedTarget']
		source_ref_p = self.owl['annotatedSource']
		man_assert_p = self.obo['ECO_0000218']
		cur_inf_p = self.obo['ECO_0000205']
		syntax_type = self.syntax_ns['type']
		axiom_ref = self.owl['Axiom']

		ret_dict = {}
		for s, p in g.subject_predicates(self.has_db_xref_p):
			axiom_res = g.resource(s)
			assert axiom_res.value(syntax_type).identifier == axiom_ref
			orpha_code = self.uri_str_to_orpha_code(axiom_res.value(source_ref_p).identifier.toPython())
			x_ref_code = axiom_res.value(target_ref_p).toPython()
			raw_link_type = (axiom_res.value(man_assert_p) or axiom_res.value(cur_inf_p)).toPython()
			if raw_link_type == 'Attributed' or raw_link_type == 'Index term':
				continue
			link_type = self.get_link_type_from_str(raw_link_type)
			assert_type = ORPHA_MANUAL_ASSERT if axiom_res.value(man_assert_p) is not None else ORPHA_CURINF_ASSERT
			dict_list_add(orpha_code, (x_ref_code, link_type, assert_type), ret_dict)
		return ret_dict


	def get_parent_info(self, res, uri_str):
		"""
		Returns:
			dict: {'IS_A': [], 'PART_OF': []}
		"""
		ret_dict = {'IS_A': [], 'PART_OF': []}
		for o in res.objects(RDFS.subClassOf):
			if isinstance(o.identifier, rdf.URIRef):
				ret_dict['IS_A'].append(self.uri_str_to_orpha_code(o.identifier.toPython()))
			elif o.value(self.on_property_p) == self.partof_ref_res:
				ret_dict['PART_OF'].append(self.uri_str_to_orpha_code(o.value(self.come_value_from_p).identifier))
		return ret_dict


	def before_read_orpha_owl(self, g):
		# print(list(g.namespaces()))
		self.orphanet_ = rdf.Namespace('http://www.orpha.net/ORDO/orphanet_#')
		self.efo = rdf.Namespace('http://www.ebi.ac.uk/efo/')
		self.obo_in_owl = rdf.Namespace('http://www.geneontology.org/formats/oboInOwl#')
		self.rdfs = rdf.Namespace('http://www.w3.org/2000/01/rdf-schema#')
		self.obo = rdf.Namespace('http://purl.obolibrary.org/obo/')
		self.owl = rdf.Namespace('http://www.w3.org/2002/07/owl#')
		self.syntax_ns = rdf.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')

		self.has_db_xref_p = self.obo_in_owl['hasDbXref']
		self.alt_term_p = self.efo['alternative_term']
		self.rdfs_label_p = self.rdfs['label']
		self.efo_def_p = self.efo['definition']
		self.partof_ref_res = g.resource(self.obo['BFO_0000050'])
		self.on_property_p = self.owl['onProperty']
		self.come_value_from_p = self.owl['someValuesFrom']


	def get_all_link_types(self):
		orpha_dict = self.get_orpha_dict()
		all_link_types = set()
		for orpha, info in orpha_dict.items():
			for db_xref, link_type, assert_type in info.get('DB_XREFS', []):
				all_link_types.add(link_type)
		return all_link_types


	def get_all_assert_types(self):
		orpha_dict = self.get_orpha_dict()
		all_assert_types = set()
		for orpha, info in orpha_dict.items():
			for db_xref, link_type, assert_type in info.get('DB_XREFS', []):
				all_assert_types.add(assert_type)
		return all_assert_types


	@check_load_save('orpha2omim', 'ORPHA_TO_OMIM', JSON_FILE_FORMAT)
	def get_all_orpha_to_omim(self):
		"""
		Returns:
			dict: {ORPHA_CODE: [(OMIM_CODE, ORPHA_TO_OMIM), ...]}
		"""
		orpha_dict = self.get_orpha_dict()
		ret_dict = {}
		for orpha_code, info in orpha_dict.items():
			for db_xref, link_type, assert_type in info.get('DB_XREFS', []):
				if db_xref.startswith('OMIM'):
					dict_list_add(orpha_code, (db_xref, link_type), ret_dict)
		return ret_dict


	@check_load_save('omim2orpha', 'OMIM_TO_ORPHA', JSON_FILE_FORMAT)
	def get_all_omim_to_orpha(self):
		"""
		Returns:
			dict: {OMIM_CODE: [(ORPHA_CODE, OMIM_TO_ORPHA), ...]}
		"""
		all_orpha_to_omim = self.get_all_orpha_to_omim()
		ret_dict = {}
		for orpha_code, omim_list in all_orpha_to_omim.items():
			for omim_code, link_orpha2omim in omim_list:
				link_omim2orpha = self.link2reversed.get(link_orpha2omim, '') or link_orpha2omim
				dict_list_add(omim_code, (orpha_code, link_omim2orpha), ret_dict)
		return ret_dict


	@check_load_save('obsolete_codes', 'OBSOLETE_CODES_JSON', JSON_FILE_FORMAT)
	def get_all_obsolete_codes(self):
		orpha_dict = self.get_orpha_dict()
		return [orpha_code for orpha_code, info in orpha_dict.items() if 'ORPHA:http://www.orpha.net/ORDO/ObsoleteClass' in info.get('IS_A', [])]


	def _get_qualified_type_codes(self, type_code_set):
		orpha_dict = self.get_orpha_dict()
		ret_list = []
		for orpha_code, info in orpha_dict.items():
			for type_code in type_code_set:
				if type_code in info.get('IS_A', []):
					ret_list.append(orpha_code)
		return ret_list


	def get_all_group_codes(self):
		return self._get_qualified_type_codes(self.level2types[DISORDER_GROUP_LEVEL])


	def get_all_disorder_codes(self):
		return self._get_qualified_type_codes(self.level2types[DISORDER_LEVEL])


	def get_all_subtype_codes(self):
		return self._get_qualified_type_codes(self.level2types[DISORDER_SUBTYPE_LEVEL])


	def get_disease_related_codes(self):
		return self.get_all_group_codes() + self.get_all_disorder_codes() + self.get_all_subtype_codes()


	def get_orpha_to_omim(self, keep_links='all', keep_asserts='all'):
		"""
		Args:
			keep_links (set or str): 'all' | set <= {
				'E': exact mapping - the terms and the concepts are equivalent
				'NTBT': narrower term maps to a broader term
				'BTNT': broader term maps to a narrower term
				'NTBT/E': narrower term maps to a broader term because of an exact mapping with a synonym in the target terminology
				'BTNT/E': broader term maps to a narrower term because of an exact mapping with a synonym in the target terminology
				'ND': not yet decided/unable to decide
				'W': incorrect mapping (two different concepts)
				None: link type is not provided
				'Specific code': The term has its own code in the ICD10
				'Inclusion term': The term is included under a ICD10 category and has not its own code
			}
			remove_asserts (set or str): 'all' | set <= {ORPHA_MANUAL_ASSERT, ORPHA_CURINF_ASSERT, None}
		Returns:
			dict: {orpha_code: [omim1, omim2]}
		"""
		if isinstance(keep_links, str):
			assert keep_links == 'all'
			keep_links = {'E', 'NTBT', 'BTNT', 'NTBT/E', 'ND', 'Specific code', 'Inclusion term', 'W', None}
		if isinstance(keep_asserts, str):
			assert keep_asserts == 'all'
			keep_asserts = {ORPHA_MANUAL_ASSERT, ORPHA_CURINF_ASSERT, None}
		orpha_dict = self.get_orpha_dict()
		ret_dict = {}
		for orpha, info in orpha_dict.items():
			for db_xref, link_type, assert_type in info.get('DB_XREFS', []):
				if db_xref.startswith('OMIM') and link_type in keep_links and assert_type in keep_asserts:
					dict_list_add(orpha, db_xref, ret_dict)
		return ret_dict


	def get_omim_to_orpha_list(self, keep_links='all', keep_asserts='all'):
		"""
		Args:
			keep_links (set or str)
			keep_asserts (set or str)
		Returns:
			dict: {omim_code: [orphaCode1, orphaCode2, ...]}
		"""
		if not isinstance(keep_links, str):
			keep_links = {self.link2reversed.get(link_omim2orpha, '') or link_omim2orpha for link_omim2orpha in keep_links}
		omim2orphas = reverse_dict_list(self.get_orpha_to_omim(keep_links, keep_asserts))
		return {omim_code: orpha_codes for omim_code, orpha_codes in omim2orphas.items() if len(orpha_codes) > 0}


	def statistic(self):
		import json
		from collections import Counter
		from core.reader.hpo_reader import HPOReader
		from core.explainer.explainer import Explainer
		from core.utils.constant import PHELIST_REDUCE
		hpo_reader = HPOReader()
		explainer = Explainer()
		# self.get_cns_orpha_dict()
		orpha_dict = self.get_orpha_dict()
		print('orpha_dict size:', len(orpha_dict))    # 14506
		disorder_related_code_set = set(self.get_disease_related_codes())
		print('Disorder related codes:', len(disorder_related_code_set)) # 9226
		print('Group of disorders:', len(self.get_all_group_codes()))  # 2122
		print('Disorders:', len(self.get_all_disorder_codes())) # 6124
		print('Subtype of disorders:', len(self.get_all_subtype_codes())) # 980
		obsolete_code_set = set(self.get_all_obsolete_codes())
		print('All obsolete codes:', len(obsolete_code_set)) # 1046
		orpha_codes_from_hpo = {dis_code for dis_code in hpo_reader.get_dis_list() if dis_code.startswith('ORPHA')}
		orpha_code_from_hpo_exists = orpha_codes_from_hpo & disorder_related_code_set
		orpha_code_from_hpo_obsolete = orpha_codes_from_hpo & obsolete_code_set
		print('Disease from hpo exists: {}/{}'.format(len(orpha_code_from_hpo_exists), len(orpha_codes_from_hpo))) # 3418/3771
		print('Disease from hpo obsolete: {}/{}'.format(len(orpha_code_from_hpo_obsolete), len(orpha_codes_from_hpo))) # 11/3771
		print('Disease from hpo left:', orpha_codes_from_hpo - orpha_code_from_hpo_exists - orpha_code_from_hpo_obsolete)

		# print('ORPHA:276238' in orpha_dict, 'ORPHA:276238' in )

		print('all link type:', self.get_all_link_types())
		print('all omim link type:', Counter([link_type for orpha_code, omims in self.get_all_orpha_to_omim().items() for omim_code, link_type in omims]))
		print('all assert type:', self.get_all_assert_types())
		print('Number of IS_A:\n', Counter([len(info.get('IS_A', [])) for code, info in orpha_dict.items()]).most_common())
		# for code, info in orpha_dict.items():
		# 	parents = info.get('IS_A', [])
		# 	if len(parents) > 2:
		# 		print(explainer.add_cns_info(code), '-> parents:', explainer.add_cns_info(parents))

		orpha2omim = self.get_orpha_to_omim()
		print('orpha2omim:', len(orpha2omim))  # 8456
		orpha2omim = self.get_orpha_to_omim({'E'})   # exact match
		print('orpha2omim exactly match:', len(orpha2omim))  # 3533
		orpha_to_multi_omim = {orpha: omimList for orpha, omimList in orpha2omim.items() if len(omimList) > 1}
		print('orpha_to_multi_omim', len(orpha_to_multi_omim), orpha_to_multi_omim)  # 33

		omim_to_orpha_list = self.get_omim_to_orpha_list()
		print('omim_to_orpha_list:', len(omim_to_orpha_list))  # 10193
		omim_to_orpha_list = self.get_omim_to_orpha_list({'E'})
		print('omim_to_orpha_list exactly match:', len(omim_to_orpha_list))  # 3563
		omim_to_multi_orpha = {omim: orpha_list for omim, orpha_list in omim_to_orpha_list.items() if len(orpha_list) > 1}
		print('omim_to_multi_orpha', len(omim_to_multi_orpha), omim_to_multi_orpha)  # 7

		dis_to_hpo = hpo_reader.get_dis_to_hpo_dict(PHELIST_REDUCE)
		total_equal, same_phenotype_count = 0, 0
		for dis_code in dis_to_hpo:
			if dis_code in omim_to_orpha_list:
				omim_code, orpha_code = dis_code, omim_to_orpha_list[dis_code][0]
				if orpha_code not in dis_to_hpo:
					continue
				total_equal += 1
				omim_phenotypes, orpha_phenotypes = sorted(dis_to_hpo[omim_code]), sorted(dis_to_hpo[orpha_code])
				if omim_phenotypes == orpha_phenotypes:
					same_phenotype_count += 1
				if total_equal < 3:
					print('{} = {}'.format(omim_code, orpha_code))
					print(omim_code, len(omim_phenotypes), explainer.add_cns_info(omim_phenotypes))
					print(orpha_code, len(orpha_phenotypes), explainer.add_cns_info(orpha_phenotypes))
		print(f'All equal pair = {total_equal}; Pairs of same phenotypes = {same_phenotype_count}')



		omim_to_orpha_list = self.get_omim_to_orpha_list({'E'})
		omim_not_in_orpha = []
		for dis_code in hpo_reader.get_dis_list():
			if not dis_code.startswith('OMIM'):
				continue
			if dis_code not in omim_to_orpha_list:
				omim_not_in_orpha.append(dis_code)
		print('hpo omim not matched by orphanet:', len(omim_not_in_orpha)) # 4186

		json.dump(
			explainer.add_cns_info(self.get_orpha_slice_dict()),
			open(os.path.join(self.PREPROCESS_FOLDER, 'orpha_slice_dict_with_cns_anno.json'), 'w'),
			indent=2, ensure_ascii=False
		)
		json.dump(
			explainer.add_cns_info(self.get_orpha_dict()),
			open(os.path.join(self.PREPROCESS_FOLDER, 'orpha_dict_with_cns_anno.json'), 'w'),
			indent=2, ensure_ascii=False
		)

		json.dump(
			explainer.add_cns_info(self.get_all_group_codes()),
			open(os.path.join(self.PREPROCESS_FOLDER, 'group_codes.json'), 'w'),
			indent=2, ensure_ascii=False
		)
		json.dump(
			explainer.add_cns_info(self.get_all_disorder_codes()),
			open(os.path.join(self.PREPROCESS_FOLDER, 'disorder_codes.json'), 'w'),
			indent=2, ensure_ascii=False
		)
		json.dump(
			explainer.add_cns_info(self.get_all_subtype_codes()),
			open(os.path.join(self.PREPROCESS_FOLDER, 'subtype_codes.json'), 'w'),
			indent=2, ensure_ascii=False
		)


if __name__ == '__main__':
	reader = OrphanetReader()
	reader.statistic()




