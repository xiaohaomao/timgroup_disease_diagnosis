import os
import numpy as np

from core.utils.constant import DATA_PATH, PHELIST_REDUCE, PHELIST_ANCESTOR, JSON_FILE_FORMAT, MODEL_PATH
from core.utils.utils import check_load_save, data_to_01_matrix
from core.reader.hpo_reader import HPOReader


class HPOICCalculator(object):
	def __init__(self, hpo_reader=HPOReader()):
		self.hpo_reader = hpo_reader
		self.HPO_IC_JSON = os.path.join(MODEL_PATH, hpo_reader.name, 'HPOICCalculator', 'IC.json')
		self.IC = None


	@check_load_save('IC', 'HPO_IC_JSON', JSON_FILE_FORMAT)
	def get_IC_dict(self):
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		self.dis2hpo = self.hpo_reader.get_dis_to_hpo_dict(phe_list_mode=PHELIST_REDUCE)
		self.hpo2dis = self.hpo_reader.get_hpo_to_dis_dict(phe_list_mode=PHELIST_REDUCE)
		hpo_to_dis_extend = {}
		self.extend_hpo_to_dis('HP:0000001', hpo_to_dis_extend)
		dis_count = len(self.dis2hpo)
		IC = {hpo_code: -np.log(len(dis_set)/dis_count) for hpo_code, dis_set in hpo_to_dis_extend.items()}    #
		return IC


	def extend_hpo_to_dis(self, code, hpo_to_dis_extend):
		if code in hpo_to_dis_extend:
			return hpo_to_dis_extend[code]
		hpo_to_dis_extend[code] = set(self.hpo2dis.get(code, []))
		for childCode in self.hpo_dict[code].get('CHILD', []):
			hpo_to_dis_extend[code].update(self.extend_hpo_to_dis(childCode, hpo_to_dis_extend))
		return hpo_to_dis_extend[code]


def get_hpo_IC_dict(hpo_reader=HPOReader(), default_IC=np.inf):
	"""
	Returns:
		dict: {HPO_CODE: IC, ...}
	"""
	def set_default_IC(IC_dict, default_IC):
		if default_IC == np.inf:
			return
		for hpo in IC_dict:
			if IC_dict[hpo] == np.inf:
				IC_dict[hpo] = default_IC
	IC_dict = HPOICCalculator(hpo_reader).get_IC_dict()
	set_default_IC(IC_dict, default_IC)
	return IC_dict


def get_hpo_IC_vec(hpo_reader=HPOReader(), default_IC=np.inf):
	"""
	Returns:
		np.ndarray: shape=(hpo_num,)
	"""
	hpo_list = HPOReader().get_hpo_list()
	hpo_to_IC = get_hpo_IC_dict(hpo_reader, default_IC)
	IC_vec = np.array([hpo_to_IC[hpo_code] for hpo_code in hpo_list])
	return IC_vec


def get_dis_IC_vec(hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR):
	HPO_NUM, DIS_NUM = hpo_reader.get_hpo_num(), hpo_reader.get_dis_num()
	IC_dict = get_hpo_IC_dict(hpo_reader)  # {HPO_CODE: IC, ...}
	IC_vec = np.array([IC_dict[hpo_code] for hpo_code in hpo_reader.get_hpo_list()])
	dis_to_hpo_int_list = hpo_reader.get_dis_int_to_hpo_int(phe_list_mode)
	dis_IC_vec = data_to_01_matrix([dis_to_hpo_int_list[i] for i in range(DIS_NUM)], HPO_NUM).dot(IC_vec).flatten()
	return dis_IC_vec


def get_dis_IC_dict(hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR):
	"""
	Returns:
		dict: {DIS_CODE: IC}
	"""
	dis_IC_vec = get_dis_IC_vec(hpo_reader, phe_list_mode)
	dis_list = hpo_reader.get_dis_list()
	return {dis_list[i]: dis_IC_vec[i] for i in range(hpo_reader.get_dis_num())}


if '__name__' == '__init__':
	pass
