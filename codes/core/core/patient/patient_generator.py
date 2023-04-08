from core.reader.hpo_reader import HPOReader
from core.utils.utils import check_load_save, dict_list_add, unique_list, split_path, delete_redundacy, slice_list_with_keep_set


class PatientGenerator(object):
	def __init__(self, hpo_reader=HPOReader()):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		self.hpo_reader = hpo_reader
		self.old_map_new_hpo = self.hpo_reader.get_old_map_new_hpo_dict()
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()
		self.all_dis_set = set(self.hpo_reader.get_dis_list())


	def get_patients(self, *args, **kwargs):
		"""
		Returns:
			list: [patient, ...]; patient=[[hpo_code, ...], [dis_code, ...]]
		"""
		raise NotImplementedError


	def get_upatients(self, *args, **kwargs):
		"""
		Returns:
			list: [upatient, ...], upatient=[hpo1, hpo2]
		"""
		raise NotImplementedError


	def filter_pa_list_with_exist_dis(self, pa_list):
		"""
		Returns:
			list: [patient, ...]; patient=[[hpo_code, ...], [dis_code, ...]]
		"""
		new_pa_list = []
		for hpo_list, dis_list in pa_list:
			newdis_list = self.process_pa_dis_list(dis_list)
			if newdis_list:
				new_pa_list.append([hpo_list, newdis_list])
		return new_pa_list


	def filter_pa_list_with_process_hpo_list(self, pa_list):
		new_pa_list = []
		for hpo_list, dis_list in pa_list:
			hpo_list = self.process_pa_hpo_list(hpo_list)
			if hpo_list:
				new_pa_list.append([hpo_list, dis_list])
		return new_pa_list


	def process_pa_dis_list(self, pa_dis_list):
		return slice_list_with_keep_set(pa_dis_list, self.all_dis_set)


	def process_pa_hpo_list(self, pa_hpo_list, reduce=True, logger=None):
		"""old -> new; delete redundacy
		Returns:
			list: pa_hpo_list
		"""
		pa_hpo_list = self.hpo_list_old_to_new(pa_hpo_list, logger)
		if reduce:
			pa_hpo_list = delete_redundacy(pa_hpo_list, self.hpo_reader.get_slice_hpo_dict())
		return unique_list(pa_hpo_list)


	def hpo_list_old_to_new(self, hpo_list, logger=None):
		"""old->new
		"""
		new_hpo_list = []
		for hpo_code in hpo_list:
			if hpo_code not in self.hpo_dict:
				if hpo_code in self.old_map_new_hpo:
					new_code = self.old_map_new_hpo[hpo_code]
					new_hpo_list.append(new_code)
					if logger: logger.info('{}(old) -> {}(new)'.format(hpo_code, new_code))
				else:
					if logger: logger.info('delete {}'.format(hpo_code))
			else:
				new_hpo_list.append(hpo_code)
		return new_hpo_list


	def diseases_from_all_sources(self, dis_codes, sources):
		if len(dis_codes) == 0:
			return False
		source_to_contain = {s: False for s in sources}
		for dis_code in dis_codes:
			for s in sources:
				if dis_code.startswith(s):
					source_to_contain[s] = True
		source_num = sum(source_to_contain.values())
		if source_num == 0:
			raise RuntimeError('Wrong disease codes:', dis_codes)
		return source_num == len(sources)


if __name__ == '__main__':
	pg = PatientGenerator()
	pass

