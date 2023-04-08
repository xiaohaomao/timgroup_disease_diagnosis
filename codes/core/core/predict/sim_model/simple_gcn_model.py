from core.reader.hpo_reader import HPOReader
from core.predict.model import SparseVecModel
from core.utils.constant import PHELIST_ANCESTOR, ADJ_MAT_AROUND, ADJ_MAT_BROTHER, PHELIST_REDUCE, VEC_TYPE_0_1
from core.utils.utils import get_brother_adj_mat, get_around_adj_mat, item_list_to_rank_list, sparse_row_normalize
from core.helper.data.data_helper import DataHelper, data_to_01_matrix

import numpy as np
import scipy.sparse as sp
import heapq

class SimpleGCNModel(SparseVecModel):
	def __init__(self, hpo_reader, phe_list_mode, adj_mat_mode, adj_order_num, model_name):
		super(SimpleGCNModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'SimpleGCNModel' if model_name is None else model_name
		self.hpo_reader = hpo_reader
		self.phe_list_mode = phe_list_mode
		self.adj_mat_mode = adj_mat_mode
		self.adj_order_num = adj_order_num
		self.adj_mat = None
		self.hpo_dis_mat = None


	def train(self):
		self.adj_mat = self.get_adj_mat(self.adj_mat_mode, self.adj_order_num)
		self.hpo_dis_mat = DataHelper().get_train_X(PHELIST_ANCESTOR, VEC_TYPE_0_1, sparse=True, dtype=np.float32).transpose()


	def get_adj_mat(self, adj_mat_mode, adj_order_num):
		if adj_mat_mode == ADJ_MAT_AROUND:
			adj_mat = get_around_adj_mat(self.hpo_reader.get_hpo_adj_mat(), adj_order_num, np.float32)
		elif adj_mat_mode == ADJ_MAT_BROTHER:
			adj_mat = get_brother_adj_mat(self.hpo_reader.get_hpo_num(), self.hpo_reader.get_hpo_int_dict(), adj_order_num, np.float32)
		else:
			assert False
		adj_mat = sparse_row_normalize(adj_mat)
		return adj_mat


	def cal_score(self, phe_list):
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return (phe_matrix * self.adj_mat * self.hpo_dis_mat).A.flatten()


	def phe_list_to_matrix(self, phe_list):
		rank_lists = [item_list_to_rank_list(phe_list, self.hpo_map_rank)]
		return data_to_01_matrix(rank_lists, self.HPO_CODE_NUMBER, dtype=np.float32)   # (1, hpo_num)


	def cal_scores(self, phe_lists):
		phe_matrix = self.phe_lists_to_matrix(phe_lists)
		return (phe_matrix * self.adj_mat * self.hpo_dis_mat).A


	def phe_lists_to_matrix(self, phe_lists):
		rank_lists = [item_list_to_rank_list(phe_list, self.hpo_map_rank) for phe_list in phe_lists]
		return data_to_01_matrix(rank_lists, self.HPO_CODE_NUMBER, dtype=np.float32)   # (1, hpo_num)


	def query_many(self, phe_lists, topk=10, chunk_size=None, cpu_use=None):
		phe_lists = [self.process_query_phe_list(phe_list, self.phe_list_mode, self.hpo_dict) for phe_list in phe_lists]
		score_matrix = self.cal_scores(phe_lists)  #
		assert np.sum(np.isnan(score_matrix)) == 0  #
		score_vecs = []
		if topk == None:
			for queryId in range(len(phe_lists)):
				score_vecs.append(sorted([(self.dis_list[i], score_matrix[queryId, i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1], reverse=True))
		else:
			for queryId in range(len(phe_lists)):
				score_vecs.append(heapq.nlargest(topk, [(self.dis_list[i], score_matrix[queryId, i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item: item[1]))
		return score_vecs


def generate_simple_gcn_model(hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, adj_mat_mode=ADJ_MAT_AROUND, adj_order_num=0, model_name=None):
	"""
	Returns:
		SimpleGCNModel
	"""
	model = SimpleGCNModel(hpo_reader, phe_list_mode, adj_mat_mode, adj_order_num, model_name)
	model.train()
	return model


class SimpleGCN2Model(SimpleGCNModel):
	def __init__(self, hpo_reader, phe_list_mode, adj_mat_mode, adj_order_num, model_name):
		super(SimpleGCN2Model, self).__init__(hpo_reader, phe_list_mode, adj_mat_mode, adj_order_num, model_name)
		self.name = 'SimpleGCN2Model' if model_name is None else model_name


	def get_adj_mat(self, adj_mat_mode, adj_order_num):
		adj_mat = super(SimpleGCN2Model, self).get_adj_mat(adj_mat_mode, 1)
		ret = sp.identity(self.hpo_reader.get_hpo_num(), np.float32)
		for i in range(adj_order_num):
			ret = ret * adj_mat
		return adj_mat



def generate_simple_gcn2_model(hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE, adj_mat_mode=ADJ_MAT_AROUND, adj_order_num=0, model_name=None):
	"""
	Returns:
		SimpleGCN2Model
	"""
	model = SimpleGCN2Model(hpo_reader, phe_list_mode, adj_mat_mode, adj_order_num, model_name)
	model.train()
	return model


if __name__ == '__main__':
	pass






