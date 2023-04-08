import numpy as np
from tqdm import tqdm

from core.reader.hpo_reader import HPOReader
from core.reader.omim_reader import OMIMReader
from core.reader.orphanet_reader import OrphanetReader
from core.explainer.explainer import Explainer

from scipy.sparse import csr_matrix

def show_feature_weight(w, b, row_names, col_names, X, saveFile, k=10):
	"""
	Args:
		b (np.array or float)
		X (csr_matrix or np.ndarray): 01-mat
	"""
	def add_cns(orderedList):
		return [(weight, col_names[colId]) for weight, colId in orderedList]
	if isinstance(X, csr_matrix):
		X = X.A     # np.ndarray
	if isinstance(b, float):
		b = np.array([b]*X.shape[0])

	s = ''
	s += 'mean(w) = {};\n std(w) = {};\n l2(w) = {};\n'.format(np.mean(w, axis=1)[4000:4100], np.std(w, axis=1)[4000:4100], np.sqrt(np.sum(w*w, axis=1))[4000:4100])
	s += 'mean(b) = {}; std(b) = {}; l2(b) = {};\n'.format(np.mean(b), np.std(b), np.sqrt(np.sum(b*b)))

	explainer = Explainer()
	col_names = explainer.add_hpo_info(col_names)
	row_names = explainer.addDisInfo(row_names)

	row_sum = X.sum(axis=0)
	none_have = set( np.argwhere(row_sum==0).flatten() )
	all_have = set( np.argwhere(row_sum==X.shape[0]).flatten() )

	for i in tqdm(range(w.shape[0])):
		ordered = sorted(zip(w[i], range(w.shape[1])), reverse=True)

		i_have = set(np.argwhere(X[i]>0).flatten()) # features(colId) that i have
		idont_have_oth_have = set(np.argwhere( (row_sum-X[i])>0 ).flatten()) - i_have # features(colId) that other disease have but i dont have

		i_have_list = [(weight, colId) for weight, colId in ordered if colId in i_have]
		idont_have_oth_have_list = [(weight, colId) for weight, colId in ordered if colId in idont_have_oth_have]
		none_have_list = [(weight, colId) for weight, colId in ordered if colId in none_have]
		all_have_list = [(weight, colId) for weight, colId in ordered if colId in all_have]

		s += '' \
			'{row_name}; HPO Number={dis_hpo_num}; bias={bias}: \n' \
			'\t topk {topk_list}\n' \
			'\t lastk {lastk_list}\n' \
			'\t i_have topk {i_have_topk_list}\n' \
			'\t i_have lastk {i_have_lastk_list}\n' \
			'\t idont_have_oth_have topk {idont_have_oth_have_topk_list}\n' \
			'\t idont_have_oth_have lastk {idont_have_oth_have_lastk_list}\n' \
			'\t none_have topk {none_have_topk_list}\n' \
			'\t none_have lastk {none_have_lastk_list}\n' \
			'\t all_have {all_have_list}\n\n'.format(
			row_name=row_names[i], dis_hpo_num=len(i_have), bias=b[i], topk_list=add_cns(ordered[:k]), lastk_list=add_cns(ordered[-k:]),
			i_have_topk_list=add_cns(i_have_list[:k]), i_have_lastk_list=add_cns(i_have_list[-k:]),
			idont_have_oth_have_topk_list=add_cns(idont_have_oth_have_list[:k]), idont_have_oth_have_lastk_list=add_cns(idont_have_oth_have_list[-k:]),
			none_have_topk_list=add_cns(none_have_list[:k]), none_have_lastk_list=add_cns(none_have_list[-k:]),
			all_have_list=add_cns(all_have_list)
		)
	print(s, file=open(saveFile, 'w'))


if __name__ == '__main__':
	pass








