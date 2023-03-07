
from core.utils.constant import EMBEDDING_PATH
from core.reader.hpo_reader import HPOReader
from core.utils.constant import PHELIST_ANCESTOR, PHELIST_ANCESTOR_DUP

import os
import numpy as np

def get_embed(encoder_name, phe_list_mode=PHELIST_ANCESTOR_DUP, hpo_reader=HPOReader()):
	"""
	Returns:
		np.ndarray: shape=[hpo_num, vec_size]
	"""
	embedfile = EMBEDDING_PATH+os.sep+'GloveEncoder'+os.sep+phe_list_mode+os.sep+encoder_name+os.sep+'vectors.txt'
	lines = [lineStr.split(' ') for lineStr in open(embedfile).read().splitlines()] # [['HP:0000001', 0.498999, ...], ['<unk>', -0.216727], ...]
	hpo_map_vec = {line[0]: line[1:] for line in lines}
	hpo_list = hpo_reader.get_hpo_list()

	embed_size = len(lines[0]) - 1
	hpo_num = hpo_reader.get_hpo_num()
	hpo_embed = np.zeros(shape=(hpo_num, embed_size), dtype=np.float32)
	for i, hpo_code in enumerate(hpo_list):
		hpo_embed[i] = hpo_map_vec[hpo_code] if hpo_code in hpo_map_vec else hpo_map_vec['<unk>']
	return hpo_embed


if __name__ == '__main__':
	hpo_reader = HPOReader()
	hpo_embed = get_embed('GloveEncoder_vec32_xMax10_max_iter200', PHELIST_ANCESTOR_DUP, hpo_reader)
	print(type(hpo_embed))
	hpo_map_rank = hpo_reader.get_hpo_map_rank()

	print('HP:3000074', hpo_embed[hpo_map_rank['HP:3000074']]) # unk






