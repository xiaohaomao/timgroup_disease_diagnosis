import numpy as np
from copy import deepcopy
import json, os

from core.reader.hpo_reader import HPOReader
from core.reader.rd_reader import RDReader
from core.reader.rd_filter_reader import RDFilterReader
from core.utils.utils import check_return, slice_dict_with_keep_set, slice_list_with_keep_set, slice_list_with_keep_func, unique_list
from core.utils.utils import item_list_to_rank_list, dict_change_key_value, dict_list_add, reverse_dict_list
from core.utils.constant import PHELIST_ORIGIN, PHELIST_ANCESTOR, PHELIST_REDUCE, DATA_PATH
from core.predict.calculator.ic_calculator import get_dis_IC_dict


class HPOFilterReader(object):
	def __init__(self, hpo_reader=HPOReader(), keep_dis_code_set=None, keep_dis_int_set=None,
			keep_hpo_code_set=None, keep_hpo_int_set=None, name=None):
		super(HPOFilterReader, self).__init__()
		self.name = name or 'FILTER'
		self.hpo_reader = hpo_reader
		self.init_keep_dis_code_set(keep_dis_code_set, keep_dis_int_set)
		self.init_keep_hpo_code_set(keep_hpo_code_set, keep_hpo_int_set)
		self.FILTER_HPO = self.k_hpo_set is not None and len(self.k_hpo_set) != 0
		self.FILTER_DIS = self.k_dis_set is not None and len(self.k_dis_set) != 0

		self.hpo_dict = None  # {CODE: {'NAME': .., 'IS_A': [], 'CHILD': [], ...}}
		self.slice_hpo_dict = None  #
		self.hpo_list = None  # [hpo_code1, hpo_code2, ...]
		self.hpo_map_rank = None  # {hpo_rank: hpo_code}
		self.hpo_int_dict = None  #
		self.used_hpo_list = None

		self.dis_list = None  # [disease_code1, disease_code2, ...]
		self.dis_map_rank = None  # {disease_rank: disease_code}
		self.dis_int_to_hpo_int = None
		self.hpo_int_to_dis_int = None

		self.old_to_new_hpo = None # {old_hpo: new_hpo}
		self.anno_hpo_list = None

		self.dis_num = None
		self.hpo_num = None


	def init_keep_dis_code_set(self, keep_dis_code_set, keep_dis_int_set):
		if keep_dis_code_set is not None:
			self.k_dis_set = keep_dis_code_set
		elif keep_dis_int_set is not None:
			dis_list = self.hpo_reader.get_dis_list()
			self.k_dis_set = {dis_list[dis_int] for dis_int in keep_dis_int_set}
		else:
			self.k_dis_set = None


	def init_keep_hpo_code_set(self, keep_hpo_code_set, keep_hpo_int_set):
		if keep_hpo_code_set is not None:
			self.k_hpo_set = keep_hpo_code_set
		elif keep_hpo_int_set is not None:
			hpo_list = self.hpo_reader.get_hpo_list()
			self.k_hpo_set = {hpo_list[hpo_int] for hpo_int in keep_hpo_int_set}
		else:
			self.k_hpo_set = None


	def process_hpo_dict(self, hpo_dict, k_set):
		if not self.FILTER_HPO:
			return hpo_dict
		hpo_dict = deepcopy(hpo_dict)
		hpo_dict = slice_dict_with_keep_set(hpo_dict, k_set)
		for hpo, info in hpo_dict.items():
			info['IS_A'] = slice_list_with_keep_set(info.get('IS_A', []), k_set)
			info['CHILD'] = slice_list_with_keep_set(info.get('CHILD', []), k_set)
		return hpo_dict


	def replace_k_to_v_list(self, d, k_to_new, v_to_new):
		return {k_to_new[k]: [v_to_new[v] for v in v_list] for k, v_list in d.items()}


	def replace_k_to_v_list_with_func(self, d, k_to_new_func, v_to_new_func):
		return {k_to_new_func(k): [v_to_new_func(v) for v in v_list] for k, v_list in d.items()}


	@check_return('hpo_dict')
	def get_hpo_dict(self):
		hpo_dict = self.hpo_reader.get_hpo_dict()
		return self.process_hpo_dict(hpo_dict, self.k_hpo_set)


	@check_return('slice_hpo_dict')
	def get_slice_hpo_dict(self):
		hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		return self.process_hpo_dict(hpo_dict, self.k_hpo_set)


	@check_return('hpo_num')
	def get_hpo_num(self):
		return len(self.get_hpo_dict())


	@check_return('dis_num')
	def get_dis_num(self):
		return len(self.get_dis_to_hpo_dict())


	@check_return('hpo_list')
	def get_hpo_list(self):
		hpo_list = self.hpo_reader.get_hpo_list()
		if not self.FILTER_HPO:
			return hpo_list
		return slice_list_with_keep_set(hpo_list, self.k_hpo_set)


	@check_return('used_hpo_list')
	def get_used_hpo_list(self):
		dis_to_hpo = self.get_dis_to_hpo_dict(PHELIST_ANCESTOR)
		return unique_list([hpo for hpo_list in dis_to_hpo.values() for hpo in hpo_list])


	@check_return('hpo_map_rank')
	def get_hpo_map_rank(self):
		if not self.FILTER_HPO:
			return self.hpo_reader.get_hpo_map_rank()
		hpo_list = self.get_hpo_list()
		return {hpo: i for i, hpo in enumerate(hpo_list)}


	@check_return('old_to_new_hpo')
	def get_old_map_new_hpo_dict(self):
		return self.hpo_reader.get_old_map_new_hpo_dict()


	@check_return('hpo_int_dict')
	def get_hpo_int_dict(self):
		hpo_dict = self.get_hpo_dict()
		hpo_map_rank = self.get_hpo_map_rank()
		hpo_int_dict = {  #
			hpo_map_rank[hpo]:{
				'IS_A': item_list_to_rank_list(info_dict.get('IS_A', []), hpo_map_rank),
				'CHILD': item_list_to_rank_list(info_dict.get('CHILD', []), hpo_map_rank)
			} for hpo, info_dict in hpo_dict.items()
		}
		return hpo_int_dict


	def process_dis_to_hpo(self, dis2hpo, k_dis, k_hpo):
		dis2hpo = deepcopy(dis2hpo)
		if self.FILTER_DIS:
			dis2hpo = slice_dict_with_keep_set(dis2hpo, k_dis)
		if self.FILTER_HPO:
			for dis, hpo_list in dis2hpo.items():
				dis2hpo[dis] = slice_list_with_keep_set(hpo_list, k_hpo)
		return dis2hpo


	def get_dis_to_hpo_dict(self, phe_list_mode=PHELIST_ORIGIN):
		dis2hpo = self.hpo_reader.get_dis_to_hpo_dict(phe_list_mode)
		return self.process_dis_to_hpo(dis2hpo, self.k_dis_set, self.k_hpo_set)


	def get_dis_int_to_hpo_int(self, phe_list_mode=PHELIST_ORIGIN):
		dis2hpo = self.get_dis_to_hpo_dict(phe_list_mode)
		return dict_change_key_value(dis2hpo, self.get_dis_map_rank(), self.get_hpo_map_rank())


	def process_hpo_to_dis(self, hpo2dis, k_hpo, k_dis):
		hpo2dis = deepcopy(hpo2dis)
		if self.FILTER_HPO:
			hpo2dis = slice_dict_with_keep_set(hpo2dis, k_hpo)
		if self.FILTER_DIS:
			for hpo, dis_list in hpo2dis.items():
				hpo2dis[hpo] = slice_list_with_keep_set(dis_list, k_dis)
		return hpo2dis


	def get_hpo_to_dis_dict(self, phe_list_mode=PHELIST_ORIGIN):
		hpo2dis = self.hpo_reader.get_hpo_to_dis_dict(phe_list_mode)
		return self.process_hpo_to_dis(hpo2dis, self.k_hpo_set, self.k_dis_set)


	def get_hpo_int_to_dis_int(self, phe_list_mode=PHELIST_ORIGIN):
		hpo2dis = self.get_hpo_to_dis_dict(phe_list_mode)
		return dict_change_key_value(hpo2dis, self.get_hpo_map_rank(), self.get_dis_map_rank())


	def process_dis_to_hpo_prob(self, dis_to_hpo_prob, k_dis, k_hpo):
		dis_to_hpo_prob = deepcopy(dis_to_hpo_prob)
		if self.FILTER_DIS:
			dis_to_hpo_prob = slice_dict_with_keep_set(dis_to_hpo_prob, k_dis)
		if self.FILTER_HPO:
			for dis, hpo_prob_list in dis_to_hpo_prob.items():
				dis_to_hpo_prob[dis] = slice_list_with_keep_func(hpo_prob_list, lambda item: item[0] in k_hpo)
		return dis_to_hpo_prob


	def get_dis_int_to_hpo_int_prob(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):
		dis_to_hpo_prob = self.get_dis_to_hpo_prob(mode, default_prob, phe_list_mode)
		dis_map_rank, hpo_map_rank = self.get_dis_map_rank(), self.get_hpo_map_rank()
		ret_dict = {}
		for dis_code, hpo_prob_list in dis_to_hpo_prob.items():
			ret_dict[dis_map_rank[dis_code]] = [[hpo_map_rank[hpo_code], prob] for hpo_code, prob in hpo_prob_list]
		return ret_dict


	def get_dis_to_hpo_raw_prob(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: dict: {dis_code: [[hpo_code, prob], ...]}; prob = 'HP:xxx' or float or None
		"""
		dis_to_hpo_prob_raw = self.hpo_reader.get_dis_to_hpo_raw_prob(phe_list_mode)
		return self.process_dis_to_hpo_prob(dis_to_hpo_prob_raw, self.k_dis_set, self.k_hpo_set)


	def hpo2freq(self, hpo_code):
		return self.hpo_reader.hpo2freq(hpo_code)


	def get_dis_to_hpo_prob(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):
		"""
		Args:
			default_prob (float or None)
		"""
		dis_to_hpo_prob = self.hpo_reader.get_dis_to_hpo_prob(mode=mode, default_prob=default_prob, phe_list_mode=phe_list_mode)
		return self.process_dis_to_hpo_prob(dis_to_hpo_prob, self.k_dis_set, self.k_hpo_set)


	def get_dis_to_hpo_prob_dict(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {dis_code: {hpo_code: prob}}
		"""
		dis_to_hpo_prob = self.get_dis_to_hpo_prob(mode=mode, default_prob=default_prob, phe_list_mode=phe_list_mode)
		return {dis: {hpo: prob for hpo, prob in hpo_prob_list} for dis, hpo_prob_list in dis_to_hpo_prob.items()}


	def get_boqa_anno_tab_rows(self, default_prob):
		rows, col_names = self.hpo_reader.get_boqa_anno_tab_rows(default_prob)
		name2rank = {name:i for i, name in enumerate(col_names)}
		new_rows = []
		for row in rows:
			dis_code = row[name2rank['DB']] + ':' + row[name2rank['DB_OBJECT_ID']]
			hpo_code = row[name2rank['HPO_CODE']]
			if self.FILTER_DIS and dis_code not in self.k_dis_set:
				continue
			if self.FILTER_HPO and hpo_code not in self.k_hpo_set:
				continue
			new_rows.append(row)
		return new_rows, col_names


	@check_return('dis_list')
	def get_dis_list(self):
		dis_list = self.hpo_reader.get_dis_list()
		if not self.FILTER_DIS:
			return dis_list
		return slice_list_with_keep_set(dis_list, self.k_dis_set)


	@check_return('dis_map_rank')
	def get_dis_map_rank(self):
		if not self.FILTER_DIS:
			return self.hpo_reader.get_dis_map_rank()
		dis_list = self.get_dis_list()
		return {dis_code: i for i, dis_code in enumerate(dis_list)}


	@check_return('anno_hpo_list')
	def get_anno_hpo_list(self):
		anno_hpo_list = self.hpo_reader.get_anno_hpo_list()
		if self.FILTER_HPO:
			anno_hpo_list = [hpo for hpo in anno_hpo_list if hpo in self.k_hpo_set]
		return anno_hpo_list


	def get_dis_to_gene_symbols(self):
		"""
		Returns:
			dict: {dis_code: [gene_symbol, ...]}
		"""
		dis_to_gene_symbols = self.hpo_reader.get_dis_to_gene_symbols()
		return slice_dict_with_keep_set(dis_to_gene_symbols, self.k_dis_set)


def get_phenomizer_dis():
	dis_codes = json.load(open(os.path.join(DATA_PATH, 'raw', 'phenomizer_sample_100', 'phenomizer_omim_orpha.json'))) # 7965
	keep_dis_set = set(HPOReader().get_dis_list())
	dis_codes = slice_list_with_keep_set(dis_codes,keep_dis_set) # 7503
	return dis_codes

#
class HPOFilterDatasetReader(HPOFilterReader):
	def __init__(self, hpo_reader=HPOReader(), keep_dnames=None, rm_no_use_hpo=False, name=None):

		keep_dnames = sorted(keep_dnames or ['OMIM', 'ORPHA', 'CCRD', 'DECIPHER'])


		name = name or '_'.join(keep_dnames)
		all_dis_codes = hpo_reader.get_dis_list()
		keep_dis_code_set = set()
		for dis_code in all_dis_codes:
			for dname in keep_dnames:
				if dis_code.startswith(dname+':'):
					keep_dis_code_set.add(dis_code)
					break
		if 'PHENOMIZERDIS' in keep_dnames:
			keep_dis_code_set.update(get_phenomizer_dis())
		keep_hpo_code_set = None
		dis_to_hpo = hpo_reader.get_dis_to_hpo_dict()
		if rm_no_use_hpo:
			keep_hpo_code_set = {hpo for dis_code in keep_dis_code_set for hpo in dis_to_hpo[dis_code]}
		super(HPOFilterDatasetReader, self).__init__(hpo_reader, keep_dis_code_set=keep_dis_code_set, keep_hpo_code_set=keep_hpo_code_set, name=name)


def get_hpo_num_slice_reader(geq_hpo_num, phe_list_mode=PHELIST_ANCESTOR):
	dis2hpo = HPOReader().get_dis_to_hpo_dict(phe_list_mode)
	return HPOFilterReader(keep_dis_code_set={dis_code for dis_code, hpo_list in dis2hpo.items() if len(hpo_list) >= geq_hpo_num})


def get_IC_slice_reader(geqIC, phe_list_mode=PHELIST_ANCESTOR):
	dis_IC_dict = get_dis_IC_dict(phe_list_mode=phe_list_mode)
	return HPOFilterReader(keep_dis_code_set={dis_code for dis_code, IC in dis_IC_dict.items() if IC >= geqIC})


class HPOIntegratedDatasetReader(HPOFilterDatasetReader):
	def __init__(self, hpo_reader=HPOReader(), keep_dnames=None, rm_no_use_hpo=False, name=None):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterReader)
		"""
		super(HPOIntegratedDatasetReader, self).__init__(hpo_reader, keep_dnames, rm_no_use_hpo, name)
		self.name = 'INTEGRATE_' + self.name

		self.rd_reader = RDReader()
		self.source2rd = self.rd_reader.get_source_to_rd()

		self.rd_list = None


	def combine_hpo_lists(self, hpo_lists):
		"""
		Args:
			hpo_lists (list): [hpo_list, hpo_list, ...]
		Returns:
			list: [hpo1, hpo2, ...]
		"""
		return list({hpo for hpo_list in hpo_lists for hpo in hpo_list})


	def combine_hpo_prob_lists(self, hpo_prob_lists, default_prob):
		"""Average over all probs; prob will be ignored if prob == None
		Args:
			hpo_prob_lists (list): [hpo_prob_list, ...], hpo_prob_list = [[hpo_code, prob], ...]
		Returns:
			list: [[hpo_code, prob], ...]
		"""
		def combine_probs(probs):
			probs = [p for p in probs if p is not None]
			if probs:
				return float(np.mean(probs))
			return default_prob

		hpo2probs = {} # {hpo: [prob1, prob2, ...]}
		for hpo_prob_list in hpo_prob_lists:
			for hpo, prob in hpo_prob_list:
				dict_list_add(hpo, prob, hpo2probs)
		return [[hpo, combine_probs(probs)] for hpo, probs in hpo2probs.items()]


	def combine_hpo_raw_prob_lists(self, hpo_raw_prob_lists):
		"""Average over all probs; prob will be ignored if prob == None
		Args:
			hpo_prob_lists (list): [hpo_prob_list, ...], hpo_prob_list = [[hpo_code, prob], ...]
		Returns:
			list: [[hpo_code, prob], ...]
		"""
		def raw_prob_to_freq(raw_prob):
			if type(raw_prob) == str and raw_prob.startswith('HP:'):
				return self.hpo2freq(raw_prob)[0]
			assert type(raw_prob) == float
			return raw_prob
		def combine_probs(probs):
			probs = [p for p in probs if p is not None]
			if len(probs) == 0:
				return None
			elif len(probs) == 1:
				return probs[0]
			elif len(probs) > 1:
				return float(np.mean([raw_prob_to_freq(p) for p in probs]))
			assert False
		hpo2probs = {}  # {hpo: [prob1, prob2, ...]}
		for hpo_prob_list in hpo_raw_prob_lists:
			for hpo, prob in hpo_prob_list:
				dict_list_add(hpo, prob, hpo2probs)
		return [[hpo, combine_probs(probs)] for hpo, probs in hpo2probs.items()]


	@check_return('rd_list')
	def get_dis_list(self):
		dis_list = super(HPOIntegratedDatasetReader, self).get_dis_list()
		return sorted(list({self.source2rd[dis_code] for dis_code in dis_list}))


	def get_dis_to_gene_symbols(self):
		dis_to_gene_symbols = super(HPOIntegratedDatasetReader, self).get_dis_to_gene_symbols()
		rd2dis = {}
		for dis in dis_to_gene_symbols:
			dict_list_add(self.source2rd[dis], dis, rd2dis)
		return {rd: np.unique([g for dis in dis_codes for g in dis_to_gene_symbols[dis]]).tolist() for rd, dis_codes in rd2dis.items()}


	def get_dis_to_hpo_dict(self, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {rd_code: [hpo_code1, ...]}
		"""
		dis2hpo = super(HPOIntegratedDatasetReader, self).get_dis_to_hpo_dict(phe_list_mode)
		rd2dis = {}
		for dis in dis2hpo:
			dict_list_add(self.source2rd[dis], dis, rd2dis)
		return {rd: self.combine_hpo_lists([dis2hpo[dis] for dis in dis_codes]) for rd, dis_codes in rd2dis.items()}


	def get_hpo_to_dis_dict(self, phe_list_mode=PHELIST_ORIGIN):
		dis_to_hpo = self.get_dis_to_hpo_dict(phe_list_mode)
		return reverse_dict_list(dis_to_hpo)


	def get_dis_to_hpo_raw_prob(self, phe_list_mode=PHELIST_ORIGIN):
		dis_to_hpo_raw_prob = super(HPOIntegratedDatasetReader, self).get_dis_to_hpo_raw_prob(phe_list_mode)
		rd2dis = {}
		for dis in dis_to_hpo_raw_prob:
			dict_list_add(self.source2rd[dis], dis, rd2dis)
		return {rd:self.combine_hpo_raw_prob_lists([dis_to_hpo_raw_prob[dis] for dis in dis_codes]) for rd, dis_codes in rd2dis.items()}


	def get_dis_to_hpo_prob(self, mode=1, default_prob=1.0, phe_list_mode=PHELIST_ORIGIN):
		"""
		Returns:
			dict: {rd_code: [[hpo_code, prob], ...]}
		"""
		dis_to_hpo_prob = super(HPOIntegratedDatasetReader, self).get_dis_to_hpo_prob(
			mode=mode, default_prob=None, phe_list_mode=phe_list_mode)
		rd2dis = {}
		for dis in dis_to_hpo_prob:
			dict_list_add(self.source2rd[dis], dis, rd2dis)
		return {rd: self.combine_hpo_prob_lists([dis_to_hpo_prob[dis] for dis in dis_codes], default_prob) for rd, dis_codes in rd2dis.items()}


	def get_boqa_anno_tab_rows(self, default_prob):
		rows, col_names = super(HPOIntegratedDatasetReader, self).get_boqa_anno_tab_rows(default_prob)
		name2rank = {name:i for i, name in enumerate(col_names)}
		rd_to_hpo_prob_dict = self.get_dis_to_hpo_prob_dict(mode=1, default_prob=default_prob)
		rd_hpo_set = set()
		new_rows = []
		for row in rows:
			dis_code = row[name2rank['DB']] + ':' + row[name2rank['DB_OBJECT_ID']]
			hpo_code = row[name2rank['HPO_CODE']]
			rd_code = self.source2rd[dis_code]
			if (rd_code, hpo_code) in rd_hpo_set:
				continue
			rd_hpo_set.add((rd_code, hpo_code))
			prob = rd_to_hpo_prob_dict[rd_code][hpo_code]
			assert prob is None or prob > 0
			freq_modifier = str(prob * 100) + '%' if prob is not None else ''
			row = deepcopy(row)
			row[name2rank['DB']] = 'RD'
			row[name2rank['DB_OBJECT_ID']] = rd_code.split(':').pop()
			row[name2rank['FREQUENCY_MODIFIER']] = freq_modifier
			row[name2rank['FREQUENCY']] = freq_modifier
			new_rows.append(row)
		return new_rows, col_names


if __name__ == '__main__':


	pass
