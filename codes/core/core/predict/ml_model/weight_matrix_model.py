from core.utils.utils import data_to_01_dense_matrix
from core.predict.model import Model
from core.reader.hpo_reader import HPOReader
from core.utils.constant import PHELIST_ANCESTOR

import numpy as np
import heapq

class WeightMatrixModel(Model):
	def __init__(self, hpo_reader, w, row_names, col_names, model_name=None):
		"""
		"""
		super(WeightMatrixModel, self).__init__()
		self.name = 'WeightMatrixModel' if model_name is None else model_name
		self.w = w
		self.row_names = row_names
		self.col_names = col_names
		self.row_name_to_idx = {name: i for i, name in enumerate(row_names)}
		self.col_name_to_idx = {name: i for i, name in enumerate(col_names)}
		self.row_num, self.col_num = self.w.shape
		self.hpo_reader = hpo_reader


	def phe_list_to_idx_list(self, phe_list):
		return [self.col_name_to_idx[hpo] for hpo in phe_list]


class SimpleWMModel(WeightMatrixModel):
	def __init__(self, hpo_reader, w, row_names, col_names, phe_list_mode, model_name=None):
		super(SimpleWMModel, self).__init__(hpo_reader, w, row_names, col_names)
		self.name = 'SimpleWMModel' if model_name is None else model_name
		self.phe_list_mode = phe_list_mode
		self.hpo_dict = self.hpo_reader.get_slice_hpo_dict()


	def query(self, phe_list, topk=10):
		phe_list = self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict)
		X = data_to_01_dense_matrix([self.phe_list_to_idx_list(phe_list)], self.col_num)
		score_vec = self.w.dot(X.T).flatten()
		assert np.sum(np.isnan(score_vec)) == 0  #
		if topk == None:
			return sorted([(self.row_names[i], score_vec[i]) for i in range(self.row_num)], key=lambda item: item[1], reverse=True)
		return heapq.nlargest(topk, [(self.row_names[i], score_vec[i]) for i in range(self.row_num)], key=lambda item: item[1])


	def query_many(self, phe_lists, topk=10, chunk_size=None, cpu_use=None):
		phe_lists = [self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict) for phe_list in phe_lists]
		X = data_to_01_dense_matrix([self.phe_list_to_idx_list(phe_list) for phe_list in phe_lists], self.col_num)
		score_matrix = X.dot(self.w.T)    # shape=(sample_num, row_num)
		score_vecs = []
		if topk == None:
			for queryId in range(len(phe_lists)):
				score_vecs.append(sorted([(self.row_names[i], score_matrix[queryId, i]) for i in range(self.row_num)], key=lambda item: item[1], reverse=True))
		else:
			for queryId in range(len(phe_lists)):
				score_vecs.append(heapq.nlargest(topk, [(self.row_names[i], score_matrix[queryId, i]) for i in range(self.row_num)], key=lambda item: item[1]))
		return score_vecs


def generate_simple_wm_model(w, row_names, col_names, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, model_name=None):
	"""
	Returns:
		SimpleWMModel
	"""
	return SimpleWMModel(hpo_reader, w, row_names, col_names, phe_list_mode, model_name)



if __name__ == '__main__':
	from core.feature.lr import LRFeatureSelector
	hpo_reader = HPOReader()
	w, row_names, col_names = LRFeatureSelector(hpo_reader, 'LRFeatureSelector_Bias0_C0.001').get_W()
	model = generate_simple_wm_model(w, row_names, col_names, hpo_reader)
	print(model.query(['HP:0000741', 'HP:0000726', 'HP:0000248', 'HP:0000369', 'HP:0000316', 'HP:0000463'], topk=20))
	pass








