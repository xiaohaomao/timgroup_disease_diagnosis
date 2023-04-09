import os
import shutil
import json
import string
import random

from core.utils.utils import get_file_list

class FileManager(object):
	def __init__(self, folder):
		self.SAVE_FOLDER = folder
		os.makedirs(folder, exist_ok=True)
		self.ID_TO_PATH_FOLDER = self.SAVE_FOLDER + os.sep + 'IdToPath'
		os.makedirs(self.ID_TO_PATH_FOLDER, exist_ok=True)

		self.ID_TO_PATH_JSON = self.SAVE_FOLDER + os.sep + 'id_to_path.json'
		self.id_to_path = self.load()


	def get_path(self, id):
		return self.id_to_path.get(id, None)


	def add_id(self, id):
		if id in self.id_to_path:
			return self.id_to_path[id]
		folder_name = ''.join(random.sample(string.ascii_letters + string.digits, 32))
		path = self.SAVE_FOLDER + os.sep + folder_name
		json.dump({id: path}, open('{}{}{}.json'.format(self.ID_TO_PATH_FOLDER, os.sep, folder_name), 'w'))
		return path


	def combine(self):
		json.dump(self.id_to_path, open(self.ID_TO_PATH_JSON, 'w'), indent=2, ensure_ascii=False)
		shutil.rmtree(self.ID_TO_PATH_FOLDER)
		os.makedirs(self.ID_TO_PATH_FOLDER)


	def load(self):
		ret = {}
		if os.path.exists(self.ID_TO_PATH_JSON):
			ret = json.load(open(self.ID_TO_PATH_JSON))
		for json_path in get_file_list(self.ID_TO_PATH_FOLDER, lambda path: path.endswith('.json')):
			ret.update(json.load(open(json_path)))
		self.check_dup_folder(ret)
		return ret


	def check_dup_folder(self, id_to_path):
		assert len(id_to_path) == len(set(id_to_path.values()))


if __name__ == '__main__':
	pass
