import itertools
from core.helper.hyper.para_searcher import ParaSearcher


class ParaGridSearcher(ParaSearcher):
	def __init__(self, key_to_values, history_list=None):
		"""
		Args:
			key_to_values (dict): {k: [v1, v2, ...]}
		"""
		super(ParaGridSearcher, self).__init__(key_to_values, history_list)
		self.key_list, self.value_lists = zip(*key_to_values.items())


	def iterator(self):
		for v_list in itertools.product(*self.value_lists):
			para_dict = {k: v for k, v in zip(self.key_list, v_list)}
			id = self.para_dict_to_id(para_dict)
			if self.id_in_history(id):
				continue
			self.add_history_id(id)
			yield para_dict



if __name__ == '__main__':
	pass








