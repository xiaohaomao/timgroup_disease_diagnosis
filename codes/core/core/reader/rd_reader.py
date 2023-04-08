import os
import json
from collections import Counter
from copy import deepcopy

from core.utils.constant import DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL, DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import reverse_dict, unique_list, dict_list_add, get_logger, delete_logger, get_all_ancestors_for_many
from core.utils.utils import check_load, check_load_save, check_return, reverse_dict_list, slice_list_with_keep_set
from core.reader.orphanet_reader import OrphanetReader
from core.reader.omim_reader import OMIMReader
from core.reader.ccrd_reader import CCRDReader
from core.reader.hpo_reader import HPOReader

class RDReader(object):
	def __init__(self):
		self.PREPROCESS_FOLDER = os.path.join(DATA_PATH, 'preprocess', 'knowledge', 'disease-mix')
		os.makedirs(self.PREPROCESS_FOLDER, exist_ok=True)
		self.RD_DICT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'rd_dict.json')
		self.rd_dict = None
		self.RD_LIST_JSON = os.path.join(self.PREPROCESS_FOLDER, 'rd_list.json')
		self.rd_list = None
		self.RD_MAP_RANK_JSON = os.path.join(self.PREPROCESS_FOLDER, 'rd_map_rank.json')
		self.rd_map_rank = None
		self.source_code_to_rd_code = None
		self.rd_code_to_source_codes = None
		self.rd_num = None
		self.source_list = None
		self.rd_dict_with_name = None
		self.RD_DICT_WITH_NAME_JSON = os.path.join(self.PREPROCESS_FOLDER, 'rd_dict_with_name.json')

		self.ORPHA_CODE_DISCARD_JSON = os.path.join(self.PREPROCESS_FOLDER, 'discard_orpha.json')
		self.DISEASE_CNS_DICT_JSON = os.path.join(self.PREPROCESS_FOLDER, 'rd_cns_dict.json')

		self.level_order = [DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
		self.level2rank = {level: i for i, level in enumerate(self.level_order)}

		self.omim_reader = OMIMReader()
		self.orpha_reader = OrphanetReader()
		self.ccrd_reader = CCRDReader()
		self.hpo_reader = HPOReader()


	def get_upper_level(self, level):
		if level == DISORDER_LEVEL:
			return DISORDER_GROUP_LEVEL, [DISORDER_GROUP_LEVEL]
		elif level == DISORDER_GROUP_LEVEL:
			return DISORDER_GROUP_LEVEL, [DISORDER_GROUP_LEVEL]
		elif level == DISORDER_SUBTYPE_LEVEL:
			return DISORDER_LEVEL, [DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
		else:
			raise RuntimeError('Unknown level: {}'.format(level))


	def get_lower_level(self, level):
		if level == DISORDER_LEVEL:
			return DISORDER_SUBTYPE_LEVEL, [DISORDER_SUBTYPE_LEVEL]
		elif level == DISORDER_SUBTYPE_LEVEL:
			return DISORDER_SUBTYPE_LEVEL, [DISORDER_SUBTYPE_LEVEL]
		elif level == DISORDER_GROUP_LEVEL:
			return DISORDER_LEVEL, [DISORDER_GROUP_LEVEL, DISORDER_LEVEL, DISORDER_SUBTYPE_LEVEL]
		else:
			raise RuntimeError('Unknown level: {}'.format(level))


	def make_rd_code(self, id):
		return f'RD:{id}'


	def _process_orpha_dict(self, orpha_dict, mark=''):
		"""
		Returns:
			dict: {
				ORPHA_CODE: {
					'LEVEL': LEVEL,
					IS_A: [ORPHA_CODE, ...]
					CHILD: [ORPHA_CODE, ...]
				}
			}
		"""
		print('into _process_orpha_dict')
		# from core.explainer import Explainer
		level2types = OrphanetReader().level2types
		type2level = {t:level for level, types in level2types.items() for t in types}
		ret_dict, discard_dict = {}, {}
		for orpha_code, info in orpha_dict.items():
			level = None
			for type_code, lev in type2level.items():
				if type_code in info.get('IS_A', []):
					assert level is None
					level = lev
			if level is None:
				discard_dict[orpha_code] = info
				continue
			ret_dict[orpha_code] = {
				'LEVEL': level,
				'IS_A': unique_list([code for code in info['IS_A'] if code not in type2level] + info.get('PART_OF', []))
			}

		child_parent_rel_set = set()
		for orpha_code, info in ret_dict.items():
			for parent_code in info.get('IS_A', []):
				child_parent_rel_set.add((info['LEVEL'], ret_dict[parent_code]['LEVEL']))
		print('All child -> parent:', child_parent_rel_set)

		json.dump(discard_dict, open(self.ORPHA_CODE_DISCARD_JSON, 'w'), indent=2, ensure_ascii=False)
		print(f'Discard orpha code: {len(discard_dict)}; Keep Orpha Code: {len(ret_dict)}')
		return ret_dict


	def _init_rd_dict(self):
		print('into _init_rd_dict')
		pp_orpha_dict = self._process_orpha_dict(self.orpha_reader.get_orpha_dict())
		# orpha_code -> rd_code
		source_code_to_rd_code = {}
		rd_num = 1
		for orpha_code in pp_orpha_dict:
			source_code_to_rd_code[orpha_code] = self.make_rd_code(rd_num); rd_num += 1

		# process orpha
		rd_dict = {}
		for orpha_code, orpha_info in pp_orpha_dict.items():
			rd_dict[source_code_to_rd_code[orpha_code]] = {
				'SOURCE_CODES':[orpha_code],
				'IS_A':[source_code_to_rd_code[orpha_code] for orpha_code in orpha_info.get('IS_A', [])],
				'CHILD':[source_code_to_rd_code[orpha_code] for orpha_code in orpha_info.get('CHILD', [])],
				'LEVEL':orpha_info['LEVEL'],
			}
		return rd_dict, source_code_to_rd_code, list(pp_orpha_dict.keys())


	def _process_source_mapping(self, rd_dict, all_source_in_codes, source_in_to_out, source_code_to_rd_code,
			default_level=DISORDER_LEVEL, mark='', keep_e=True):
		# process exact mapping
		exact_map_out = set()
		for in_code in all_source_in_codes:
			rd_code = source_code_to_rd_code[in_code]
			for out_code, link_type in source_in_to_out.get(in_code, []):
				if link_type == 'E' or link_type is None:
					source_code_to_rd_code[out_code] = rd_code
					rd_dict[rd_code]['SOURCE_CODES'].append(out_code)
					dict_list_add('CANDIDATE_LEVEL', rd_dict[rd_code]['LEVEL'], rd_dict[rd_code])
					exact_map_out.add(out_code)

		# process not exact orpha-omim mapping
		CONFILICT_LEVEL_TXT = os.path.join(self.PREPROCESS_FOLDER, f'{mark}_conflict_level.txt')
		logger = get_logger('conflict_level', CONFILICT_LEVEL_TXT, mode='w')
		for in_code in all_source_in_codes:
			rd_code = source_code_to_rd_code[in_code]
			for out_code, link_type in source_in_to_out.get(in_code, []):
				if out_code in exact_map_out and keep_e:
					continue
				elif link_type == 'E' or link_type is None:
					continue
				elif link_type == 'BTNT' or link_type == 'BTNT/E' or link_type == 'NTBT' or link_type == 'NTBT/E':
					# prefix
					if out_code in source_code_to_rd_code:
						new_rd_code = source_code_to_rd_code[out_code]
					else:
						new_rd_code = self.make_rd_code(len(rd_dict) + 1)
						source_code_to_rd_code[out_code] = new_rd_code
					new_rd_info = rd_dict.get(new_rd_code, {})

					# internal
					if link_type == 'BTNT' or link_type == 'BTNT/E':
						dict_list_add('SOURCE_CODES', out_code, new_rd_info)
						dict_list_add('IS_A', rd_code, new_rd_info)
						level, candidates = self.get_lower_level(rd_dict[rd_code]['LEVEL'])
					else:
						assert link_type == 'NTBT' or link_type == 'NTBT/E'
						dict_list_add('SOURCE_CODES', out_code, new_rd_info)
						dict_list_add('IS_A', new_rd_code, rd_dict[rd_code])
						level, candidates = self.get_upper_level(rd_dict[rd_code]['LEVEL'])

					# postfix
					if 'LEVEL' in new_rd_info:
						if new_rd_info['LEVEL'] != level:
							dict_list_add('CANDIDATE_LEVEL', candidates, new_rd_info)
							cand_counter = Counter()
							for candidates in new_rd_info['CANDIDATE_LEVEL']:
								cand_counter.update(candidates)
							level_and_count = cand_counter.most_common()
							if len(cand_counter) != 1:
								choose_level = default_level if level_and_count[0][1] == level_and_count[1][1] else level_and_count[0][0]
								logger.info('Level Confilict: {}; {} -> {}'.format(out_code, level_and_count, choose_level))
								new_rd_info['LEVEL'] = choose_level
							else:
								new_rd_info['LEVEL'] = level_and_count[0][0]
					else:
						new_rd_info['LEVEL'] = level
						dict_list_add('CANDIDATE_LEVEL', candidates, new_rd_info)
					rd_dict[new_rd_code] = new_rd_info
				else:
					assert link_type == 'W' or link_type == 'ND'
		delete_logger(logger)


	def add_isolated_source(self, rd_dict, source_code, source_code_to_rd_code):
		new_rd_code = self.make_rd_code(len(rd_dict) + 1)
		source_code_to_rd_code[source_code] = new_rd_code
		rd_dict[new_rd_code] = {
			'SOURCE_CODES':[source_code],
			'LEVEL':DISORDER_LEVEL,
		}


	def combine_rd_codes(self, rd_codes, rd_dict, source_code_to_rd_code, default_level=DISORDER_LEVEL):
		def old_list_to_new(rd_codes, old_rd_code_set, new_rd_code):
			return unique_list([ (new_rd_code if rd_code in old_rd_code_set else rd_code) for rd_code in rd_codes])
		new_rd_code = rd_codes[0]
		new_info_dict = rd_dict[new_rd_code]

		# set info dict
		level_counter = Counter([new_info_dict['LEVEL']])
		for rd_code in rd_codes[1:]:
			old_info_dict = rd_dict[rd_code]
			new_info_dict['SOURCE_CODES'].extend(old_info_dict['SOURCE_CODES'])
			new_info_dict['IS_A'].extend(old_info_dict['IS_A'])
			new_info_dict['CHILD'].extend(old_info_dict['CHILD'])
			level_counter[old_info_dict['LEVEL']] += 1
			del rd_dict[rd_code]

		# set level
		for k in new_info_dict:
			if isinstance(new_info_dict[k], list):
				new_info_dict[k] = unique_list(new_info_dict[k])
		level_freq = level_counter.most_common()
		if len(level_freq) == 1:
			new_info_dict['LEVEL'] = level_freq[0][0]
		else:
			new_info_dict['LEVEL'] = level_freq[0][0] if level_freq[0][1] > level_freq[1][1] else default_level
			print('Level conflict in combining {}: {}; choose {}'.format(rd_codes, level_freq, new_info_dict['LEVEL']))
		rd_dict[new_rd_code] = new_info_dict

		# change pointer in other rd codes
		old_rd_code_set = set(rd_codes[1:])
		for rd_code, info in rd_dict.items():
			info['IS_A'] = old_list_to_new(info.get('IS_A', []), old_rd_code_set, new_rd_code)
			info['CHILD'] = old_list_to_new(info.get('CHILD', []), old_rd_code_set, new_rd_code)

		# set source_code_to_rd_code
		for source_code in new_info_dict['SOURCE_CODES']:
			source_code_to_rd_code[source_code] = new_rd_code


	@check_load_save('rd_dict_with_name', 'RD_DICT_WITH_NAME_JSON', JSON_FILE_FORMAT)
	def get_rd_dict_with_name(self):
		rd_dict = deepcopy(self.get_rd_dict())
		cns_dicts = [self.omim_reader.get_cns_omim(), self.orpha_reader.get_cns_orpha_dict(), self.ccrd_reader.get_ccrd_dict()]
		eng_dicts = [self.omim_reader.get_omim_dict(), self.orpha_reader.get_orpha_dict(), self.ccrd_reader.get_ccrd_dict()]
		for rd_code, info in rd_dict.items():
			cns_name, eng_name = None, None
			for source_code in info['SOURCE_CODES']:
				for cns_dict in cns_dicts:
					cns_name = cns_name or cns_dict.get(source_code, {}).get('CNS_NAME', None)
				for eng_dict in eng_dicts:
					eng_name = eng_name or eng_dict.get(source_code, {}).get('ENG_NAME', None)
			if cns_name is not None:
				info['CNS_NAME'] = cns_name
			if eng_name is not None:
				info['ENG_NAME'] = eng_name
		return rd_dict


	@check_load_save('rd_dict', 'RD_DICT_JSON', JSON_FILE_FORMAT)
	def get_rd_dict(self):
		"""
		Returns:
			dict: {
				DIS_CODE: {
					'SOURCE_CODES': [ORPHA:XXX, OMIM:XXXXXX, ...],
					'LEVEL': str,
					'IS_A': [],
					'CHILD': [],
				}
			}
		"""
		print('into get_rd_dict')
		rd_dict, source_code_to_rd_code, all_orpha_codes = self._init_rd_dict()

		orpha_to_omim = self.orpha_reader.get_all_orpha_to_omim()
		self._process_source_mapping(rd_dict, all_orpha_codes, orpha_to_omim, source_code_to_rd_code, DISORDER_LEVEL, mark='orpha_to_omim')

		orpha_to_ccrd = self.ccrd_reader.get_orpha_to_ccrd()
		self._process_source_mapping(rd_dict, all_orpha_codes, orpha_to_ccrd, source_code_to_rd_code, DISORDER_LEVEL, mark='orpha_to_ccrd', keep_e=False)
		omim_to_ccrd = self.ccrd_reader.get_omim_to_ccrd()
		self._process_source_mapping(rd_dict, all_orpha_codes, omim_to_ccrd, source_code_to_rd_code, DISORDER_LEVEL, mark='omim_to_ccrd', keep_e=False)

		# process isolated source codes
		source_codes_from_hpo = self.hpo_reader.get_dis_list()
		for source_code in source_codes_from_hpo:
			if source_code not in source_code_to_rd_code:
				self.add_isolated_source(rd_dict, source_code, source_code_to_rd_code)

		# add CHILD
		for rd_code, info in rd_dict.items():
			for parent_code in info.get('IS_A', []):
				dict_list_add('CHILD', rd_code, rd_dict[parent_code])

		# Deduplication and final check
		for rd_code, rd_info in rd_dict.items():
			rd_info['SOURCE_CODES'] = sorted(unique_list(rd_info['SOURCE_CODES']))
			rd_info['IS_A'] = sorted(unique_list(rd_info.get('IS_A', [])))
			rd_info['CHILD'] = sorted(unique_list(rd_info.get('CHILD', [])))
			if 'CANDIDATE_LEVEL' in rd_info:
				del rd_info['CANDIDATE_LEVEL']
			assert 'LEVEL' in rd_info

		# Combine RD codes
		source_to_rds = reverse_dict_list({rd_code:rd_info['SOURCE_CODES'] for rd_code, rd_info in rd_dict.items()})
		for source_code, rd_codes in source_to_rds.items():
			if len(rd_codes) > 1:
				print('Dup mapping: {} -> {}'.format(source_code, rd_codes))
				self.combine_rd_codes(rd_codes, rd_dict, source_code_to_rd_code, DISORDER_LEVEL)

		return rd_dict


	def get_filter_rd_dict(self, keep_source_codes):
		rd_dict = deepcopy(self.get_rd_dict())
		for rd, info in rd_dict.items():
			info['SOURCE_CODES'] = slice_list_with_keep_set(info['SOURCE_CODES'], keep_source_codes)
		return rd_dict


	def get_all_group_codes(self):
		return self.get_all_level_codes(DISORDER_GROUP_LEVEL)


	def get_all_disorder_codes(self):
		return self.get_all_level_codes(DISORDER_LEVEL)


	def get_all_subtype_codes(self):
		return self.get_all_level_codes(DISORDER_SUBTYPE_LEVEL)


	def get_all_level_codes(self, level):
		rd_dict = self.get_rd_dict()
		return [rd_code for rd_code, info in rd_dict.items() if info['LEVEL'] == level]


	def get_level_leaf_codes(self, level):
		def is_leaf(rd):
			for child_rd in rd_dict[rd].get('CHILD', []):
				if rd_dict[child_rd]['LEVEL'] == level:
					return False
			return True

		rd_dict = self.get_rd_dict()
		ret_rd_codes = []
		for rd, info in rd_dict.items():
			if info['LEVEL'] == level and is_leaf(rd):
				ret_rd_codes.append(rd)
		return ret_rd_codes


	def set_level_leaf_codes(self, rd_dict, level, to_level):
		rd_codes = self.get_level_leaf_codes(level)
		for rd in rd_codes:
			rd_dict[rd]['LEVEL'] = to_level
		return rd_dict


	@check_return('rd_code_to_source_codes')
	def get_rd_to_sources(self):
		"""
		Returns:
			dict: {DIS_CODE: [SOURCE_CODE1, ...]}
		"""
		rd_dict = self.get_rd_dict()
		return {rd_code: rd_info['SOURCE_CODES'] for rd_code, rd_info in rd_dict.items()}


	@check_return('source_code_to_rd_code')
	def get_source_to_rd(self):
		"""
		Returns:
			dict: {SOURCE_CODE: DIS_CODE}
		"""
		dis_to_sources = self.get_rd_to_sources()
		return {source_code: rd_code for rd_code, source_codes in dis_to_sources.items() for source_code in source_codes}


	@check_return('rd_num')
	def get_rd_num(self):
		return len(self.get_rd_dict())


	def gen_rank_json(self, item_list, item_list_json, item_map_rank_json):
		item_num = len(item_list)
		item_map_rank = {item_list[i]: i for i in range(item_num)}
		json.dump(item_list, open(item_list_json, 'w'), indent=2)
		json.dump(item_map_rank, open(item_map_rank_json, 'w'), indent=2)
		return item_list, item_map_rank


	@check_load('rd_list', 'RD_LIST_JSON', JSON_FILE_FORMAT)
	def get_rd_list(self):
		rd_list, _ = self.gen_rank_json(list(self.get_rd_dict().keys()), self.RD_LIST_JSON, self.RD_MAP_RANK_JSON)
		return rd_list


	@check_load('rd_map_rank', 'RD_MAP_RANK_JSON', JSON_FILE_FORMAT)
	def get_rd_map_rank(self):
		_, rd_map_rank = self.gen_rank_json(list(self.get_rd_dict().keys()), self.RD_LIST_JSON, self.RD_MAP_RANK_JSON)
		return rd_map_rank

	@check_return('source_list')
	def get_source_list(self):
		return list(self.get_source_to_rd().keys())


	def get_rd_to_level(self):
		rd_dict = self.get_rd_dict()
		return {rd_code: info['LEVEL'] for rd_code, info in rd_dict.items()}


	def get_source_to_level(self):
		rd_dict = self.get_rd_dict()
		return {source_code: info['LEVEL'] for _, info in rd_dict.items() for source_code in info['SOURCE_CODES']}


	def statistics(self):
		from core.explainer.explainer import Explainer
		from tqdm import tqdm
		def print_rd_with_most_child(level=DISORDER_GROUP_LEVEL, topk=50):
			from core.utils.utils import get_all_descendents
			all_level_rds = self.get_all_level_codes(level)
			all_level_rd_set = set(all_level_rds)
			rd_dict = self.get_rd_dict()
			rd_to_desc_num = {}
			for rd in tqdm(all_level_rds):
				desc_set = get_all_descendents(rd, rd_dict)
				rd_to_desc_num[rd] = len([desc_rd for desc_rd in desc_set if desc_rd in all_level_rd_set])
			rd_num_pairs = sorted(rd_to_desc_num.items(), key=lambda item:item[1], reverse=True)
			for i in range(topk):
				rd, desc_num = rd_num_pairs[i]
				print('{}; {}/{}; {}'.format(rd, desc_num, len(all_level_rds), explainer.add_cns_info(rd)))

		explainer = Explainer()

		rd_dict = self.get_rd_dict()
		print('All disease number:', self.get_rd_num())
		print('Disease level:', Counter([info['LEVEL'] for rd_code, info in rd_dict.items()]).most_common()) # [('DISORDER', 8201), ('DISORDER_SUBTYPE', 3363), ('DISORDER_GROUP', 2265)]
		print('Isolated codes:', len([rd_code for rd_code, info in rd_dict.items() if len(info.get('IS_A', [])) == 0 and len(info.get('CHILD', [])) == 0])) # 1995

		self.get_rd_to_sources()
		self.get_source_to_rd()

		json.dump(explainer.add_cns_info(rd_dict), open(self.DISEASE_CNS_DICT_JSON, 'w'), indent=2, ensure_ascii=False)
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
		json.dump(
			explainer.add_cns_info(self.get_level_leaf_codes(DISORDER_GROUP_LEVEL)),
			open(os.path.join(self.PREPROCESS_FOLDER, 'disorder_group_leaf.json'), 'w'),
			indent=2, ensure_ascii=False
		)
		print_rd_with_most_child()


	def check(self):
		from core.utils.utils import reverse_dict_list
		rd_to_sources = self.get_rd_to_sources()
		source_to_rds = reverse_dict_list(rd_to_sources)
		for source_code, rd_codes in source_to_rds.items():
			if len(rd_codes) > 2:
				print('source -> rds:', source_code, rd_codes)
		print(len(self.get_rd_dict()), len(self.get_rd_list()), len({rd_code for _, rd_code in self.get_source_to_rd().items()}))


def source_codes_to_rd_codes(source_codes, rd_reader=None):
	"""
	Args:
		source_codes (list):
	Returns:
		list: [rd_code1, rd_code2, ...]
	"""
	rd_reader = rd_reader or RDReader()
	source_to_rd = rd_reader.get_source_to_rd()



	return list({source_to_rd[source_code] for source_code in source_codes})


if __name__ == '__main__':
	reader = RDReader()
	reader.statistics()
	reader.check()
	reader.get_rd_dict_with_name()


