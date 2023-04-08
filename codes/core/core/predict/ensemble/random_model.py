import heapq
import numpy as np
import os

from core.predict.model import Model
from core.reader.hpo_reader import HPOReader
from core.utils.utils import check_load_save
from core.utils.constant import MODEL_PATH, NPY_FILE_FORMAT

class RandomModel(Model):
	def __init__(self, hpo_reader=HPOReader(), seed=None, model_name=None):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterReader)
		"""
		super(RandomModel, self).__init__()
		self.name = model_name or 'RandomModel'
		self.hpo_reader = hpo_reader
		self.seed = seed
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.dis_list = hpo_reader.get_dis_list()
		self.SAVE_FOLDER = os.path.join(MODEL_PATH, self.hpo_reader.name, 'RandomModel')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)

		self.RANDOM_SCORE_VEC_NPY = os.path.join(self.SAVE_FOLDER, 'random_scorevec_{}.npy'.format(self.seed))
		self.random_score_vec = None

		self.train()

	def train(self):
		self.get_random_score_vec()


	@check_load_save('random_score_vec', 'RANDOM_SCORE_VEC_NPY', NPY_FILE_FORMAT)
	def get_random_score_vec(self):
		rnd = np.random.RandomState(self.seed)
		return rnd.random(self.DIS_NUM)


	def query_score_vec(self, phe_list):
		"""
		Returns:
			np.ndarray: shape=(dis_num,)
		"""
		return self.random_score_vec


	def query_score_mat(self, phe_lists, chunk_size=None, cpu_use=12):
		return np.vstack([self.random_score_vec for i in phe_lists])


	def score_vec_to_result(self, score_vec, topk):
		"""
		Args:
			score_vec (np.ndarray): (dis_num,)
			topk (int or None):
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		if topk == None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item:item[1])


if __name__ == '__main__':
	from core.reader import HPOFilterDatasetReader
	hpo_reader = HPOFilterDatasetReader(keep_dnames=['OMIM'])
	model = RandomModel(hpo_reader=hpo_reader, seed=777)

