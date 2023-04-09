import random
from core.helper.hyper.para_grid_searcher import ParaSearcher

class ParaRandomSearcher(ParaSearcher):
	def __init__(self, key_to_values, history_list=None):
		"""
		Args:
			key_to_values (dict): {k: [v1, v2, ...]}
		"""
		super(ParaRandomSearcher, self).__init__(key_to_values, history_list)
		self.key_to_ranks = {k: list(range(len(v_list))) for k, v_list in key_to_values.items()}

		self.WARNING_LOOP_NUM = 100
		self.ABORT_LOOP_NUM = 10000


	def select(self):
		"""
		Returns:
			dict: {k: rank1, ...}
		"""
		return {k: random.sample(v_list, 1)[0] for k, v_list in self.key_to_values.items()}


	def next(self):
		loop_count = 0
		while True:
			loop_count += 1
			if loop_count > self.WARNING_LOOP_NUM:
				print('Warning: loop {} times for choosing next parameters'.format(loop_count))
			if loop_count > self.ABORT_LOOP_NUM:
				print('Abort: too much loop times to choose next parameter')
				break
			para_dict = self.select()
			id = self.para_dict_to_id(para_dict)
			if self.id_in_history(id):
				continue
			self.add_history_id(id)
			return para_dict
		return 'No Next Parameters'


	def iterator(self, max_iter):
		"""
		Returns:
			dict: {key: value, ...}
		"""
		assert max_iter <= self.MAX_ITER
		iter_count = 0
		while iter_count < max_iter:
			iter_count += 1
			yield self.next()


if __name__ == '__main__':
	pass
