"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.reader.hpo_reader import HPOReader
from core.utils.constant import PHELIST_ANCESTOR_DUP, PHELIST_ANCESTOR, DATA_PATH


def gentext(outfile, phe_type=PHELIST_ANCESTOR_DUP, hpo_reader=HPOReader()):
	dis_to_hpo_dict = hpo_reader.get_dis_to_hpo_dict(phe_type)
	with open(outfile, 'w') as fout:
		for dis, hpo_list in dis_to_hpo_dict.items():
			print(' '.join(hpo_list), file=fout)
	return max([len(hpo_list) for hpo_list in dis_to_hpo_dict.values()])


if __name__ == '__main__':
	max_len = gentext(DATA_PATH+'/preprocess/GloveTextAncestorDup', phe_type=PHELIST_ANCESTOR_DUP)
	print('max_len = {}'.format(max_len)) # 1546
	max_len = gentext(DATA_PATH+'/preprocess/GloveTextAncestor', phe_type=PHELIST_ANCESTOR)
	print('max_len = {}'.format(max_len)) # 553




