


class ParaSearcher(object):
	def __init__(self, key_to_values, history_list=None):
		"""
		Args:
			key_to_values (dict): {k: [v, ...], ...}
			history_list (list): [{k: v}, ...]
		"""
		self.key_to_values = key_to_values
		self.MAX_ITER = self.cal_permutation()
		self.history_id_set = set()
		if history_list is not None:
			self.add_multi_history(history_list)


	def cal_permutation(self):
		count = 1
		for k, v_list in self.key_to_values.items():
			count *= len(v_list)
		return count


	def para_dict_to_id(self, para_dict):
		return str( sorted(list(para_dict.items())) )


	def add_history_id(self, id):
		self.history_id_set.add(id)


	def add_history(self, para_dict):
		"""
		Args:
			para_dict (dict): {key: value}
		"""
		self.add_history_id(self.para_dict_to_id(para_dict))


	def add_multi_history(self, paraDicts):
		"""
		Args:
			paraDicts (list): [{key: value}, ...]
		"""
		for para_dict in paraDicts:
			self.add_history(para_dict)


	def add_hyper_tune_history(self, hyperTuneHelper):
		self.add_multi_history(hyperTuneHelper.get_para_history())


	def id_in_history(self, id):
		return id in self.history_id_set


	def in_history(self, para_dict):
		return self.para_dict_to_id(para_dict) in self.history_id_set



if __name__ == '__main__':
	pass


