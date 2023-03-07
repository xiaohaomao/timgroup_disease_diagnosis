

from lxml import etree
import json
import os

from core.utils.constant import DATA_PATH, JSON_FILE_FORMAT
from core.utils.utils import check_load_save, unique_list, timer
from core.explainer.dataset_explainer import UnLabeledDatasetExplainer
from core.draw.simpledraw import simple_dist_plot
from core.patient.patient_generator import PatientGenerator

class SCVWithHPOTarget(object):
	def __init__(self):
		print('SCVTarget init')
		self.scv_item_list = []
		self.scv_item = self.init_scv_item()
		self.in_scv = False
		self.in_trait = ''   # 'Finding'| 'Disease'
		self.tag = ''

	def start(self, tag, attrib):
		self.tag = tag
		if tag == 'ClinVarAssertion':
			self.in_scv = True
		if not self.in_scv: return
		if tag == 'ClinVarSubmissionID':
			self.scv_item['SUBMITTER'] = attrib['submitter']
		if tag == 'ClinVarAccession':
			self.scv_item['SCV_CODE'] = attrib['Acc'].strip()
		if tag == 'Trait':
			self.in_trait = attrib['Type']
		if self.in_trait != '' and tag == 'XRef':
			db = attrib.get('DB', '')
			if db == 'Human Phenotype Ontology' or db == 'HP':
				self.scv_item['TRAIT_HPO'].append(attrib['ID'].strip())

	def data(self, data):
		data = data.strip()
		if len(data) == 0: return
		if not self.in_scv: return
		if self.tag == 'ReviewStatus':
			self.scv_item['REVIEW_STATUS'] = data
		if self.tag == 'Description':
			self.scv_item['CLINICAL_SIGNIFICANCE'] = data

	def end(self, tag):
		if tag == 'ClinVarAssertion':
			self.append_scv_item()
			self.scv_item = self.init_scv_item()
			self.in_scv = False
		if tag == 'Trait':
			self.in_trait = ''

	def close(self):
		return self.scv_item_list

	def init_scv_item(self):
		return {'TRAIT_HPO': []}

	def append_scv_item(self):
		if not self.scv_item['TRAIT_HPO']: return
		self.scv_item['TRAIT_HPO'] = unique_list(self.scv_item['TRAIT_HPO'])
		self.scv_item_list.append(self.scv_item)


class RCVWithHPOTarget(object):
	def __init__(self):
		print('RCVTarget init')
		self.rcv_item_list = []
		self.rcv_item = self.init_rcv_item()
		self.in_rcv = False
		self.in_trait = ''   # 'Finding'| 'Disease'
		self.tag = ''

	def start(self, tag, attrib):
		self.tag = tag
		if tag == 'ReferenceClinVarAssertion':
			self.in_rcv = True
		if not self.in_rcv: return
		if tag == 'ClinVarAccession':
			self.rcv_item['RCV_CODE'] = attrib['Acc'].strip()
		if tag == 'Trait':
			self.in_trait = attrib['Type']
		if self.in_trait != '' and tag == 'XRef':
			db = attrib.get('DB', '')
			if db == 'Human Phenotype Ontology' or db == 'HP':
				self.rcv_item['TRAIT_HPO'].append(attrib['ID'].strip())

	def data(self, data):
		data = data.strip()
		if len(data) == 0: return
		if not self.in_rcv: return
		if self.tag == 'ReviewStatus':
			self.rcv_item['REVIEW_STATUS'] = data
		if self.tag == 'Description':
			self.rcv_item['CLINICAL_SIGNIFICANCE'] = data

	def end(self, tag):
		if tag == 'ReferenceClinVarAssertion':
			self.append_rcv_item()
			self.rcv_item = self.init_rcv_item()
			self.in_rcv = False
		if tag == 'Trait':
			self.in_trait = ''

	def close(self):
		return self.rcv_item_list

	def init_rcv_item(self):
		return {'TRAIT_HPO': []}

	def append_rcv_item(self):
		if not self.rcv_item['TRAIT_HPO']: return
		self.rcv_item['TRAIT_HPO'] = unique_list(self.rcv_item['TRAIT_HPO'])
		self.rcv_item_list.append(self.rcv_item)


class ClinvarPatientGenerator(PatientGenerator):
	def __init__(self):
		super(ClinvarPatientGenerator, self).__init__()
		self.CLINVAR_FULL_RELEASE_XML = DATA_PATH + '/raw/ClinVar/ClinVarFullRelease_2019-01.xml'
		self.OUTPUT_FOLDER = DATA_PATH + '/preprocess/patient/ClinVar'
		os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

		self.CLINVAR_SCV_ITEM_JSON = os.path.join(self.OUTPUT_FOLDER, 'scv_item_list.json')
		self.scv_item_list = None
		self.CLINVAR_SCV_UPATIENTS_JSON = os.path.join(self.OUTPUT_FOLDER, 'scv_upatients.json')
		self.scv_upatients = None

		self.CLINVAR_RCV_ITEM_JSON = os.path.join(self.OUTPUT_FOLDER, 'rcv_item_list.json')
		self.rcv_item_list = None
		self.CLINVAR_RCV_UPATIENTS_JSON = os.path.join(self.OUTPUT_FOLDER, 'rcv_upatients.json')
		self.rcv_upatients = None


	def get_upatients(self):
		return self.get_rcv_upatients()


	@timer
	@check_load_save('scv_item_list', 'CLINVAR_SCV_ITEM_JSON', JSON_FILE_FORMAT)
	def get_scv_item_list(self):
		"""
		Returns:
			list: [
				{'SCV_CODE': '', 'TRAIT_HPO': [hpo_code, ...], 'SUBMITTER': '', 'REVIEW_STATUS': '', 'CLINICAL_SIGNIFICANCE': ''}
			]
		"""
		parser = etree.XMLParser(target=SCVWithHPOTarget())
		return etree.parse(self.CLINVAR_FULL_RELEASE_XML, parser)


	@check_load_save('scv_upatients', 'CLINVAR_SCV_UPATIENTS_JSON', JSON_FILE_FORMAT)
	def get_scv_upatients(self):
		"""
		Returns:
			list: [upatient, ...], upatient=[hpo1, hpo2]
		"""
		scv_list = self.get_scv_item_list()
		upatients = []
		for pa_item in scv_list:
			pa_hpo_list = self.process_pa_hpo_list(pa_item['TRAIT_HPO'])
			if len(pa_hpo_list) == 0: continue
			upatients.append(pa_hpo_list)
		return upatients


	@timer
	@check_load_save('rcv_item_list', 'CLINVAR_RCV_ITEM_JSON', JSON_FILE_FORMAT)
	def get_rcv_item_list(self):
		"""
		Returns:
			list: [
				{'SCV_CODE': '', 'TRAIT_HPO': [hpo_code, ...], 'SUBMITTER': '', 'REVIEW_STATUS': '', 'CLINICAL_SIGNIFICANCE': ''}
			]
		"""
		parser = etree.XMLParser(target=RCVWithHPOTarget())
		return etree.parse(self.CLINVAR_FULL_RELEASE_XML, parser)


	@check_load_save('rcv_upatients', 'CLINVAR_RCV_UPATIENTS_JSON', JSON_FILE_FORMAT)
	def get_rcv_upatients(self):
		"""
		Returns:
			list: [upatient, ...], upatient=[hpo1, hpo2]
		"""
		rcv_list = self.get_rcv_item_list()
		upatients = []
		for pa_item in rcv_list:
			pa_hpo_list = self.process_pa_hpo_list(pa_item['TRAIT_HPO'])
			if len(pa_hpo_list) == 0: continue
			upatients.append(pa_hpo_list)
		return upatients


if __name__ == '__main__':
	pass
	cpg = ClinvarPatientGenerator()

	cpg.get_rcv_item_list()
	upatients = cpg.get_rcv_upatients()
	explainer = UnLabeledDatasetExplainer(upatients)

	simple_dist_plot(cpg.OUTPUT_FOLDER + '/HPO_NUMBER_RCV.jpg', [len(hpo_list) for hpo_list in upatients], 100,'HPO Number of Patient')



