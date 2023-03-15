

from core.utils.constant import EMBEDDING_PATH

import os
import numpy as np


def get_embed(encoder_name, method='SDNEEncoder'):
	"""
	Returns:
		np.ndarray: shape=[hpo_num, vec_size]
	"""
	embedfile = EMBEDDING_PATH+os.sep+method+os.sep+encoder_name+'.txt'
	lines = [lineStr.split(' ') for lineStr in open(embedfile).read().splitlines()] # [['HP:0000001', 0.498999, ...], ['<unk>', -0.216727], ...]
	vecNum, embed_size = int(lines[0][0]), int(lines[0][1])
	hpo_embed = np.zeros(shape=(vecNum, embed_size), dtype=np.float32)
	for line in lines[1:]:
		hpo_int = int(line[0])
		hpo_embed[hpo_int] = line[1:]
	return hpo_embed


if __name__ == '__main__':
	hpo_embed = get_embed('encoder[128,32]_epoch200_alpha0.000001_beta5_nu1-0.00001_nu2-0.0001')
	print(hpo_embed[0])