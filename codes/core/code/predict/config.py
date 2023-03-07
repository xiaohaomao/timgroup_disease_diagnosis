

import json
from core.utils.utils import is_jsonable

class Config(object):
	def __init__(self, d=None):
		if d is not None:
			self.assign(d)


	def __str__(self):
		return '\n'.join('%s: %s' % item for item in self.__dict__.items())


	def save(self, path, delete_unjson=False):
		if delete_unjson:
			json.dump(self.jsonable_filter(self.__dict__), open(path, 'w'), indent=2)
		else:
			json.dump(self.__dict__, open(path, 'w'), indent=2)


	def load(self, path):
		self.assign(json.load(open(path)))


	def assign(self, value_dict):
		for key in value_dict:
			if not hasattr(self, key):
				raise RuntimeError('Wrong key of Config: {}'.format(key))
			setattr(self, key, value_dict[key])


	def jsonable_filter(self, d):
		return {k:v for k, v in d.items() if is_jsonable(v)}


if __name__ == '__main__':
	print(Config({'a': 1, 'b': 2}))

