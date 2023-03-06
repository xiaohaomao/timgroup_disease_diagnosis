"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from core.reader.hpo_reader import HPOReader
from feature.pca import PCADimReductor, PCAConfig
from core.utils.utils import getDisHPOMat
from core.utils.constant import PHELIST_ANCESTOR
from tqdm import tqdm
from multiprocessing import Pool


def process(para):
	pca_config, model_name = para
	hpo_reader = HPOReader()

	X, _, _ = getDisHPOMat(hpo_reader, PHELIST_ANCESTOR, sparse=False)
	pca = PCADimReductor(model_name)
	pca.train(X, pca_config)
	return pca


def train_script():
	n_components = [0.95, 5000, 8000, 10000] # [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	paras = []
	for compoNum in n_components:
		pca_config = PCAConfig()
		pca_config.n_component = compoNum
		model_name = 'PCADimReductor_Compo{}'.format(compoNum)
		paras.append([pca_config, model_name])

	for para in tqdm(paras):
		process(para)



if __name__ == '__main__':
	train_script()
	pass