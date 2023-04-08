from scipy.sparse import csr_matrix
from core.reader.hpo_reader import HPOReader

from core.predict.model import SparseVecModel
from collections import Counter
from core.utils.utils import get_all_ancestors, item_list_to_rank_list
from core.utils.constant import PHELIST_REDUCE
from core.predict.calculator.tfidf_calculator import TFIDFCalculator

class TFIDFModel(SparseVecModel):
	def __init__(self, hpo_reader, phe_list_mode=PHELIST_REDUCE):
		"""
		Args:
			hpo_dict (dict): {CODE: {'IS_A': [], 'CHILD': []}}
			dis2hpo (dict): {DISEASE_CODE: [HPO_CODE, ...], }
		"""
		super(TFIDFModel, self).__init__(hpo_reader, phe_list_mode)
		self.name = 'TFIDFModel'


	def train(self):
		print('training...')
		self.cal_dis_vec_mat()


	def cal_dis_vec_mat(self):
		calculator = TFIDFCalculator()
		self.transformer = calculator.get_default_transformer()
		self.dis_vec_mat = calculator.get_default_dis_hpo_mat()


	def cal_score(self, phe_list):
		"""
		Args:
			phe_matrix (scipy.sparse.csr.csr_matrix): shape=[1, hpo_num]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		phe_matrix = self.phe_list_to_matrix(phe_list)
		return self.cal_dot_product(phe_matrix)


	def phe_list_to_rank_and_freq(self, phe_list):
		"""
		Returns:
			list: Rank
			list: Frequency
		"""
		counter = Counter()
		for hpo in phe_list:
			counter.update(get_all_ancestors(hpo, self.hpo_dict))
		rank_list = item_list_to_rank_list(counter.keys(), self.hpo_map_rank)
		freq_list = list(counter.values())
		return rank_list, freq_list


	def phe_list_to_matrix(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo1, hpo2]
		Returns:
			scipy.sparse.csr.csr_matrix: shape=[1, hpo_num]
		"""
		rank_list, freq_list = self.phe_list_to_rank_and_freq(phe_list)
		m = csr_matrix((freq_list, ([0]*len(rank_list), rank_list)), shape=(1, self.HPO_CODE_NUMBER))
		return self.transformer.transform(m)


def generate_model(hpo_reader=HPOReader(), phe_list_mode=PHELIST_REDUCE):
	"""
	Returns:
		TFIDFModel
	"""
	model = TFIDFModel(hpo_reader, phe_list_mode)
	model.train()
	return model

if __name__ == '__main__':
	model = generate_model()

