import numpy as np
import heapq
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from core.predict.model import Model
from core.utils.utils import item_list_to_rank_list, timer, list_find, data_to_01_matrix, sparse_element_max
from core.utils.constant import PHELIST_ANCESTOR
from core.reader.hpo_reader import HPOReader


class BayesNetBFModel(Model):
	def __init__(self, hpo_reader, alpha, cond_type='max', model_name=None):
		super(BayesNetBFModel, self).__init__()
		self.name = 'BayesNetBFModel' if model_name is None else model_name
		self.hpo_reader = hpo_reader
		self.DIS_CODE_NUMBER = hpo_reader.get_dis_num()
		self.HPO_CODE_NUMBER = hpo_reader.get_hpo_num()
		self.dis_list = hpo_reader.get_dis_list()
		self.hpo_map_rank = hpo_reader.get_hpo_map_rank()
		self.dis_int_to_hpo_int = hpo_reader.get_dis_int_to_hpo_int(PHELIST_ANCESTOR)
		self.hpo_dict = hpo_reader.get_slice_hpo_dict()

		self.alpha = alpha
		self.cond_type = cond_type
		self.EPS = 0.0001


	def train(self):
		self.G = self.get_G()
		gc = self.getgc()
		self.U = self.get_unbias_log_prob(self.G, gc)
		self.dis_maskvs = {i:self.get_maskv(self.dis_int_to_hpo_int[i]) for i in range(self.DIS_CODE_NUMBER)}
		self.dis_B = {i: self.get_bias_log_prob(self.G, gc, self.dis_maskvs[i]) for i in tqdm(range(self.DIS_CODE_NUMBER))}


	@timer
	def get_G(self):
		"""parent HPO To HPO Matrix
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num); row=parent, col=child
		"""
		hpo_int_dict = self.hpo_reader.get_hpo_int_dict()
		return data_to_01_matrix([hpo_int_dict[i].get('CHILD', []) for i in range(self.HPO_CODE_NUMBER)], self.HPO_CODE_NUMBER, np.int32)

	@timer
	def getgc(self):
		""" number of each hpo's children
		Returns:
			np.ndarray: shape=(hpo_num, 1)
		"""
		return self.G.sum(axis=1)+self.EPS

	@timer
	def get_unbias_log_prob(self, G, gc):
		"""matrix U
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num); row=parent, col=child
		"""
		return G.multiply(-np.log(gc)).asformat('csr')


	def get_bias_log_prob(self, G, gc, dis_maskv):
		"""matrix B
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num); row=parent, col=child
		"""
		D = self.get_L(dis_maskv)
		logdc = np.log(D.sum(axis=1) + gc * self.alpha)
		D1 = D.multiply(np.log(1 + self.alpha) - logdc)
		D2 = (dis_maskv.multiply(G) - D).multiply(np.log(0 + self.alpha) - logdc)
		return D1 + D2  # csr_matrix


	def get_L(self, maskv):
		"""limited parent HPO To children HPO Matrix; children must be in hpo_int_list
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num); row=parent, col=child
		"""
		return maskv.T.multiply(self.G).multiply(maskv)


	def get_maskv(self, hpo_int_list, dtype=np.bool_):
		return data_to_01_matrix([hpo_int_list], self.HPO_CODE_NUMBER, dtype=dtype).T


	def get_diags(self, hpo_int_list):
		diag = np.zeros(self.HPO_CODE_NUMBER, dtype=np.int32)
		diag[hpo_int_list] = 1
		return sp.diags(diag, format='csr')


	def cal_P(self, Q, U, B, dmaskv, qmaskv):
		"""condition log prob of each p(child|parent)
		Returns:
			csr_matrix: shape=(hpo_num, hpo_num); row=parent, col=child
		"""
		return Q.multiply(B) + (qmaskv - qmaskv.multiply(dmaskv)).multiply(Q).multiply(U)


	def p_to_log_prob(self, P, cond_type='max'):
		"""
		Returns:
			float
		"""
		if cond_type == 'max':
			return sparse_element_max(P, axis=0).sum()
		elif cond_type == 'ind': # independent
			P.data = np.log(1-np.exp(P.data))
			return np.log(1-np.exp(P.sum(axis=0))).sum()
		assert False


	@timer
	def cal_score(self, phe_list):
		"""too slow 20 seconds/per query
		"""
		phpo_int_list = item_list_to_rank_list(phe_list, self.hpo_reader.get_hpo_map_rank())
		qmaskv = self.get_maskv(phpo_int_list)
		Q = self.get_L(qmaskv)
		return [self.p_to_log_prob(self.cal_P(Q, self.U, self.dis_B[i], self.dis_maskvs[i], qmaskv)) for i in tqdm(range(self.DIS_CODE_NUMBER))]


	def query(self, phe_list, topk=10):
		if len(phe_list) == 0:
			return self.query_empty(topk)
		phe_list = self.process_query_phe_list(phe_list, PHELIST_ANCESTOR, self.hpo_dict)
		score_vec = self.cal_score(phe_list)
		assert np.sum(np.isnan(score_vec)) == 0
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_CODE_NUMBER)], key=lambda item:item[1])


def generate_model(hpo_reader=HPOReader(), alpha=0.5, cond_type='max', model_name=None):
	"""
	Returns:
		BayesNetBFModel
	"""
	model = BayesNetBFModel(hpo_reader, alpha, cond_type, model_name)
	model.train()
	return model


if __name__ == '__main__':
	model = generate_model()









