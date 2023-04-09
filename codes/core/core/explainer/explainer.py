import json
from copy import deepcopy
from core.reader import HPOReader, OMIMReader, OrphanetReader, CCRDReader, DOReader, RDReader
from core.explainer.utils import add_info

class Explainer(object):
	def __init__(self, hpo_reader=HPOReader()):
		self.hpo_reader = hpo_reader
		self.omim_reader = OMIMReader()
		self.orpha_reader = OrphanetReader()
		self.ccrd_reader = CCRDReader()
		self.do_reader = DOReader()
		self.rd_reader = RDReader()

		self.hpo_to_info = self.get_hpo_to_info()
		self.orpha_to_info = self.get_orpha_to_info()
		self.omim_to_info = self.get_omim_to_info()
		self.ccrd_to_info = self.get_ccrd_to_info()
		self.rd_to_info = self.get_rd_to_info()
		self.do_to_info = self.get_do_to_info()
		self.gene_to_info = self.get_gene_to_info()

		self.dis_to_hpo_prob_dict = self.hpo_reader.get_dis_to_hpo_prob_dict(default_prob=None)


	def explain(self):
		raise NotImplementedError


	def write_as_str(self, explain_dict):
		return json.dumps(explain_dict, indent=2, ensure_ascii=False)


	def explain_as_str(self):
		explain_dict = self.explain()
		return self.write_as_str(explain_dict)


	def explain_save_txt(self, txtpath):
		explain_dict = self.explain()
		self.write_as_txt(explain_dict, txtpath)


	def explain_save_json(self, jsonpath):
		explain_dict = self.explain()
		self.write_as_json(explain_dict, jsonpath)


	def write_as_txt(self, explain_dict, txtPath):
		print(self.write_as_str(explain_dict), file=open(txtPath, 'w'))


	def write_as_json(self, explain_dict, json_path):
		# explain_dict = self.add_cns_info(explain_dict)
		json.dump(explain_dict, open(json_path, 'w'), indent=2, ensure_ascii=False)


	def to_str_list(self, obj):
		if isinstance(obj, list):
			return [str(item) for item in obj]
		elif isinstance(obj, dict):
			return ['{}: {}'.format(k, v) for k, v in obj.items()]
		assert False


	def add_cns_info(self, obj, mode='a'):
		obj = deepcopy(obj)
		obj = self.add_hpo_info(obj, mode)
		obj = self.add_omim_info(obj, mode)
		obj = self.add_orpha_info(obj, mode)
		obj = self.add_ccrd_info(obj, mode)
		obj = self.add_rd_info(obj, mode)
		obj = self.add_gene_info(obj, mode)
		return obj


	def add_dis_cns_info(self, obj, mode='a'):
		return self.add_ccrd_info(self.add_orpha_info(self.add_omim_info(obj, mode), mode))


	def get_hpo_to_info(self):
		hpo_to_cns_info = {hpo :info['CNS_NAME'] for hpo, info in self.hpo_reader.get_chpo_dict().items() if info.get('CNS_NAME', None)}
		hpo_to_eng_info = {hpo: info['ENG_NAME'] for hpo, info in self.hpo_reader.get_hpo_dict().items() if info.get('ENG_NAME', None)}
		return dict(hpo_to_eng_info, **hpo_to_cns_info)


	def add_hpo_info(self, obj, mode='a'):
		return add_info(obj, self.hpo_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('HP:'), mode=mode)


	def get_orpha_to_info(self):
		orpha_to_cns_info = {orpha: info['CNS_NAME'] for orpha, info in self.orpha_reader.get_cns_orpha_dict().items() if info.get('CNS_NAME', None)}
		orpha_to_eng_info = {orpha: info['ENG_NAME'] for orpha, info in self.orpha_reader.get_orpha_dict().items() if info.get('ENG_NAME', None)}
		return dict(orpha_to_eng_info, **orpha_to_cns_info)


	def add_orpha_info(self, obj, mode='a'):
		return add_info(obj, self.orpha_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('ORPHA:'), mode=mode)


	def get_omim_to_info(self):
		omim_to_cns_info = {omim: info['CNS_NAME'] for omim, info in self.omim_reader.get_cns_omim().items() if info.get('CNS_NAME', None)}
		omim_to_eng_info = {omim: info['ENG_NAME'] for omim, info in self.omim_reader.get_omim_dict().items() if info.get('ENG_NAME', None)}
		return dict(omim_to_eng_info, **omim_to_cns_info)


	def add_omim_info(self, obj, mode='a'):
		return add_info(obj, self.omim_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('OMIM:'), mode=mode)


	def get_ccrd_to_info(self):
		ccrd_to_cns_info = {omim:info['CNS_NAME'] for omim, info in self.ccrd_reader.get_ccrd_dict().items() if info.get('CNS_NAME', None)}
		ccrd_to_eng_info = {omim:info['ENG_NAME'] for omim, info in self.ccrd_reader.get_ccrd_dict().items() if info.get('ENG_NAME', None)}
		return dict(ccrd_to_eng_info, **ccrd_to_cns_info)


	def add_ccrd_info(self, obj, mode='a'):
		return add_info(obj, self.ccrd_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('CCRD:'), mode=mode)


	def get_rd_to_info(self):
		rd_dict = self.rd_reader.get_rd_dict_with_name()
		rd_to_cns_info = {rd:info['CNS_NAME'] for rd, info in rd_dict.items() if info.get('CNS_NAME', None)}
		rd_to_eng_info = {rd:info['ENG_NAME'] for rd, info in rd_dict.items() if info.get('ENG_NAME', None)}
		return dict(rd_to_eng_info, **rd_to_cns_info)


	def add_rd_info(self, obj, mode='a'):
		return add_info(obj, self.rd_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('RD:'), mode=mode)


	def get_do_to_info(self):
		do_to_eng_info = {do: info['ENG_NAME'] for do, info in self.do_reader.get_do_dict().items()}
		return do_to_eng_info


	def add_do_info(self, obj, mode='a'):
		return add_info(obj, self.do_to_info, lambda tgt:isinstance(tgt, str) and tgt.startswith('DOID:'), mode=mode)


	def get_gene_to_info(self):
		return self.hpo_reader.get_gene_to_symbol()


	def add_gene_info(self, obj, mode='a'):
		return add_info(obj, self.gene_to_info, lambda tgt: isinstance(tgt, str) and tgt.startswith('EZ:'), mode=mode)



if __name__ == '__main__':
	pass







