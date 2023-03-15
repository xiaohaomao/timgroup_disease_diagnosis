
import numpy as np

from core.utils.utils import get_all_ancestors_with_dist, list_find, split_path


def get_match_impre_noise_with_dist(true_hpo_set, input_hpo_it, hpo_dict):
	"""
	Args:
		true_hpo_set (set): {hpo_code1, hpo_code2, ...}
		input_hpo_it (iterator): e.g. [hpo_code1, hpo_code2, ...]
		hpo_dict (dict):
	Returns:
		list: match hpo; [hpo_code1, ...], hpo_code1 in input_hpo_it
		list: impre hpo; [(hpo_code1, (hpo_code2, distance)), ...], hpo_code1 in input_hpo_it and hpo_code2 in true_hpo_set
		list: specific hpo; [(hpo_code1, (hpo_code2, distance)), ...], hpo_code1 in input_hpo_it and hpo_code2 in true_hpo_set
		list: other hpo; [hpo_code1, ...], hpo_code1 in input_hpo_it
	"""
	def get_ances_with_dis_dict(hpo, hpo_dict):
		if hpo in hpo_to_ances_with_dis:
			return hpo_to_ances_with_dis[hpo]
		hpo_to_ances_with_dis[hpo] = get_all_ancestors_with_dist(hpo, hpo_dict)
		return hpo_to_ances_with_dis[hpo]

	hpo_to_ances_with_dis = {}
	mat, imp, noi_spe, noi_oth = [], [], [], []
	for hpo in input_hpo_it:
		if hpo in true_hpo_set:  # match
			mat.append(hpo)
			continue
		impre_tuple = ('', np.inf)
		for t_hpo in true_hpo_set:  # imprecision
			ances_with_dis_dict = get_ances_with_dis_dict(t_hpo, hpo_dict)
			if hpo in ances_with_dis_dict and ances_with_dis_dict[hpo] < impre_tuple[1]:
				impre_tuple = (t_hpo, ances_with_dis_dict[hpo])
		if impre_tuple[1] != np.inf:
			imp.append((hpo, impre_tuple))
			continue
		ances_with_dis_dict = get_ances_with_dis_dict(hpo, hpo_dict)  # noise Specified
		noi_spe_tuple = ('', np.inf)
		for t_hpo in true_hpo_set:
			if t_hpo in ances_with_dis_dict and ances_with_dis_dict[t_hpo] < noi_spe_tuple[1]:
				noi_spe_tuple = (t_hpo, ances_with_dis_dict[t_hpo])
		if noi_spe_tuple[1] != np.inf:
			noi_spe.append((hpo, noi_spe_tuple))
			continue
		noi_oth.append(hpo)
	return mat, imp, noi_spe, noi_oth


def get_match_impre_noise_with_dist_detail(true_hpo_set, input_hpo_it, hpo_dict):
	"""
	Args:
		true_hpo_set (set): {hpo_code1, hpo_code2, ...}
		input_hpo_it (iterator): e.g. [hpo_code1, hpo_code2, ...]
		hpo_dict (dict):
	Returns:
		list: match hpo; [hpo_code1, ...], hpo_code1 in input_hpo_it
		list: impre hpo; [(hpo_code1, [(hpo_code2, distance), ...]), ...], hpo_code1 in input_hpo_it and hpo_code2 in true_hpo_set
		list: specific hpo; [(hpo_code1, [(hpo_code2, distance), ...]), ...], hpo_code1 in input_hpo_it and hpo_code2 in true_hpo_set
		list: other hpo; [hpo_code1, ...], hpo_code1 in input_hpo_it
	"""
	def get_ances_with_dis_dict(hpo, hpo_dict):
		if hpo in hpo_to_ances_with_dis:
			return hpo_to_ances_with_dis[hpo]
		hpo_to_ances_with_dis[hpo] = get_all_ancestors_with_dist(hpo, hpo_dict)
		return hpo_to_ances_with_dis[hpo]

	hpo_to_ances_with_dis = {}
	mat, imp, noi_spe, noi_oth = [], [], [], []
	for hpo in input_hpo_it:
		if hpo in true_hpo_set:  # match
			mat.append(hpo)
			continue
		impre_tuples = []
		for t_hpo in true_hpo_set:  # imprecision
			ances_with_dis_dict = get_ances_with_dis_dict(t_hpo, hpo_dict)
			if hpo in ances_with_dis_dict:
				impre_tuples.append((t_hpo, ances_with_dis_dict[hpo]))
		if impre_tuples:
			imp.append((hpo, impre_tuples))
			continue
		ances_with_dis_dict = get_ances_with_dis_dict(hpo, hpo_dict)  # noise Specified
		noi_spe_tuples = []
		for t_hpo in true_hpo_set:
			if t_hpo in ances_with_dis_dict:
				noi_spe_tuples.append((t_hpo, ances_with_dis_dict[t_hpo]))
		if noi_spe_tuples:
			noi_spe.append((hpo, noi_spe_tuples))
			continue
		noi_oth.append(hpo)
	return mat, imp, noi_spe, noi_oth


def get_match_impre_noise(true_hpo_set, input_hpo_it, hpo_dict):
	mat, imp, noi_spe, noi_oth = get_match_impre_noise_with_dist(true_hpo_set, input_hpo_it, hpo_dict)
	imp = [hpo for hpo, tupl in imp]
	noi_spe = [hpo for hpo, tupl in noi_spe]
	return mat, imp, noi_spe, noi_oth


def add_info(obj, tgt2info, tgt_filter, mode='a'):
	"""
	Args:
		mode (str): 'a'|'w'
	"""
	if tgt_filter(obj) and obj in tgt2info:
		return str(obj) + '-' + str(tgt2info[obj]) if mode == 'a' else str(tgt2info[obj])
	if isinstance(obj, tuple):
		return tuple([add_info(obj[i], tgt2info, tgt_filter, mode) for i in range(len(obj))])
	if isinstance(obj, list):
		return [add_info(obj[i], tgt2info, tgt_filter, mode) for i in range(len(obj))]
	if isinstance(obj, set):
		return set([add_info(item, tgt2info, tgt_filter, mode) for item in obj])
	if isinstance(obj, np.ndarray):
		return np.array([add_info(obj[i], tgt2info, tgt_filter, mode) for i in range(len(obj))])
	if isinstance(obj, dict):
		return {add_info(k, tgt2info, tgt_filter, mode):add_info(v, tgt2info, tgt_filter, mode) for k, v in obj.items()}
	return obj


def obj2str(obj, depth=0, tab='  '):
	return obj_to_str_with_max_depth(obj, depth, tab, max_depth=None)


def obj_to_str_with_max_depth(obj, depth=0, tab='  ', max_depth=None):
	"""Note: list of list will be flattened
	Args:
		max_depth (int or None): be single line when depth >= max_depth
	"""
	if max_depth is not None and depth >= max_depth:
		return tab*depth+str(obj)+'\n'
	if isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, set) or isinstance(obj, np.ndarray):
		return ''.join([obj_to_str_with_max_depth(child_obj, depth, tab, max_depth) for child_obj in obj])
	if isinstance(obj, dict):
		return ''.join([
			'{}{}'.format(tab*depth+str(k)+':'+'\n', obj_to_str_with_max_depth(child_obj, depth+1, tab, max_depth))
			for k, child_obj in obj.items()
		])
	return tab*depth+str(obj)+'\n'


def add_tab(s, tab='  '):
	return tab + s.replace('\n', '\n'+tab)


if __name__ == '__main__':
	pass