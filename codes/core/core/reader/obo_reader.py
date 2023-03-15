

import re

from core.utils.utils import dict_list_add


class OBOReader(object):
	def __init__(self):
		self.handle_func = self.get_handle_func()
		self.key_map = {
			'name':'ENG_NAME', 'is_a':'IS_A', 'def':'ENG_DEF', 'synonym':'SYNONYM', 'creation_date':'CREATED_DATE',
			'created_by':'CREATED_BY', 'alt_id':'ALT_ID', 'xref':'XREF', 'comment':'COMMENT', 'replaced_by':'REPLACED_BY',
			'subset': 'SUBSET', 'url': 'URL'
		}


	def load(self, path, child_key=True):
		"""
		Returns:
			dict: {id: info_dict}; info_dict={'ENG_NAME': '', 'IS_A': '', 'CHILD': '', ...}
		"""
		with open(path) as f:
			return self.loads(f.read())


	def loads(self, s, child_key=True):
		ret_dict = {}
		raw_list = s.split('[Term]')[1:]
		for raw_str in raw_list:
			code, info_dict = self.handle_raw_item(raw_str)
			if code != None:
				ret_dict[code] = info_dict

		if child_key:
			self.add_child_key(ret_dict)
		return ret_dict


	def handle_raw_item(self, raw_str):
		lines = raw_str.strip().split('\n')
		code = None
		info_dict = {}
		for line in lines:
			key, value = self.get_key_value(line)
			if key == 'is_obsolete' and value == 'true':    #
				return None, None
			if key == 'id':
				code = value
			elif key in self.handle_func:
				self.handle_func[key](key, value, info_dict, self.key_map)
		return code, info_dict


	def add_child_key(self, d):
		for code in d:
			for p_code in d[code].get('IS_A', []):
				dict_list_add('CHILD', code, d[p_code])


	def get_handle_func(self):
		def hsv(k, v, info_dict, key_map):    # handle single value
			info_dict[key_map[k]] = v
		def hlv(k, v, info_dict, key_map):    # handle list value
			dict_list_add(key_map[k], v, info_dict)
		def handle_isa_value(k, v, info_dict, key_map):
			v = v.split('!')[0].strip()
			hlv(k, v, info_dict, key_map)
		def handle_syn_value(k, v, info_dict, key_map):
			v = re.match('^"(.+)" .* \[.*\]( \{.*\})?$', v).group(1)
			hlv(k, v, info_dict, key_map)
		return {
			'name':hsv, 'is_a':handle_isa_value, 'def':hsv, 'synonym':handle_syn_value, 'creation_date':hsv,
			'created_by':hsv, 'alt_id':hlv, 'xref':hlv, 'comment':hsv, 'replaced_by':hsv, 'subset': hlv,
			'url': hlv,
		}


	def get_key_value(self, line):
		key, value = line.split(':', maxsplit=1)
		return key.strip(), value.strip()








