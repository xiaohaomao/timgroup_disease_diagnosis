# encoding: UTF-8
from core.reader.hpo_reader import HPOReader
from core.reader.hpo_filter_reader import HPOFilterReader, HPOFilterDatasetReader, HPOIntegratedDatasetReader
from core.reader.omim_reader import OMIMReader
from core.reader.orphanet_reader import OrphanetReader
from core.reader.ccrd_reader import CCRDReader
from core.reader.do_reader import DOReader
from core.reader.umls_reader import UMLSReader
from core.reader.rd_reader import RDReader, source_codes_to_rd_codes
from core.reader.rd_filter_reader import RDFilterReader


def get_omim_codes_and_name(hpo_reader=None, omim_reader=None):
	hpo_reader = hpo_reader or HPOReader()
	omim_reader = omim_reader or OMIMReader()
	dis2name_hpo_provide = hpo_reader.get_dis_to_name()
	cns_dict = omim_reader.get_cns_omim()
	code_dict = omim_reader.get_omim_dict()
	dis2eng, dis2cns = {}, {}
	all_codes = set(list(cns_dict.keys()) + list(code_dict.keys()))
	all_codes.update([dis_code for dis_code in dis2name_hpo_provide if dis_code.startswith('OMIM:')])
	for code in all_codes:
		eng_name = code_dict.get(code, {}).get('ENG_NAME', '') or dis2name_hpo_provide.get(code, '')
		if eng_name:
			dis2eng[code] = eng_name
		cns_name = cns_dict.get(code, {}).get('CNS_NAME', '')
		if cns_name:
			dis2cns[code] = cns_name
	return all_codes, dis2eng, dis2cns


def get_orpha_codes_and_name(hpo_reader=None, orpha_reader=None):
	hpo_reader = hpo_reader or HPOReader()
	source_reader = orpha_reader or OrphanetReader()
	dis2name_hpo_provide = hpo_reader.get_dis_to_name()
	cns_dict = source_reader.get_cns_orpha_dict()
	code_dict = source_reader.get_orpha_dict()
	dis2eng, dis2cns = {}, {}
	all_codes = set(list(cns_dict.keys()) + list(code_dict.keys()))
	all_codes.update([dis_code for dis_code in dis2name_hpo_provide if dis_code.startswith('ORPHA:')])
	for code in all_codes:
		eng_name = code_dict.get(code, {}).get('ENG_NAME', '') or dis2name_hpo_provide.get(code, '')
		if eng_name:
			dis2eng[code] = eng_name
		cns_name = cns_dict.get(code, {}).get('CNS_NAME', '')
		if cns_name:
			dis2cns[code] = cns_name
	return all_codes, dis2eng, dis2cns


def get_ccrd_codes_and_names(ccrd_reader=None):
	source_reader = ccrd_reader or CCRDReader()
	ccrd_dict = source_reader.get_ccrd_dict()
	all_codes = list(ccrd_dict.keys())
	dis2eng = {code: info['ENG_NAME'] for code, info in ccrd_dict.items()}
	dis2cns = {code:info['CNS_NAME'] for code, info in ccrd_dict.items()}
	return all_codes, dis2eng, dis2cns


def get_rd_codes_and_names(hpo_reader=None, rd_reader=None, omim_reader=None, orpha_reader=None, ccrd_reader=None):
	def get_codes_match_prefix(codes, prefix):
		return [code for code in codes if code.startswith(prefix)]
	def combine_cns_eng(args):
		all_codes, code2eng, code2cns = args
		return {code: (code2cns.get(code, ''), code2eng.get(code, '')) for code in all_codes}
	def set_if_not_empty(d, k, v):
		if v:
			d[k] = v
	order = ['CCRD', 'ORPHA', 'OMIM']
	prefix_to_cns_eng_dict = {
		'OMIM': combine_cns_eng(get_omim_codes_and_name(hpo_reader, omim_reader)),
		'ORPHA': combine_cns_eng(get_orpha_codes_and_name(hpo_reader, orpha_reader)),
		'CCRD': combine_cns_eng(get_ccrd_codes_and_names(ccrd_reader))
	}
	rd_dict = rd_reader.get_rd_dict()
	rd2eng, rd2cns = {}, {}
	for rd, info in rd_dict.items():
		eng_name, cns_name = '', ''
		cand_eng_name = ''
		for prefix in order:
			code_to_cns_eng = prefix_to_cns_eng_dict[prefix]
			codes = get_codes_match_prefix(info['SOURCE_CODES'], prefix)
			for code in codes:
				if cns_name:
					break
				cns_name, eng_name = code_to_cns_eng.get(code, ('', ''))
				cand_eng_name = cand_eng_name or eng_name
		if not cns_name:
			eng_name = cand_eng_name
		set_if_not_empty(rd2eng, rd, eng_name)
		set_if_not_empty(rd2cns, rd, cns_name)
	return list(rd_dict.keys()), rd2eng, rd2cns


if __name__ == '__main__':
	pass


