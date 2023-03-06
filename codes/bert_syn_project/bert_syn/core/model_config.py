
import copy
import json

class Config(object):
	def __init__(self):
		pass


	def __str__(self):
		return json.dumps(self.__dict__, indent=2)


	def save(self, path):
		json.dump(self.__dict__, open(path, 'w'), indent=2)


	def load(self, path):
		self.from_dict(json.load(open(path)))


	def from_dict(self, config_dict):
		for k, v in config_dict.items():
			setattr(self, k, v)


	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		return copy.deepcopy(self.__dict__)


if __name__ == '__main__':
	pass
