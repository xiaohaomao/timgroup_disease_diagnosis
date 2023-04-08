import os
import numpy as np
from core.predict.model import SklearnModel
from sklearn.mixture import GaussianMixture
import joblib
import scipy.sparse as sp

from core.predict.config import Config
from core.utils.utils import timer
from core.utils.constant import MODEL_PATH, PREDICT_MODE, PHELIST_ANCESTOR, VEC_TYPE_0_1
from core.reader.hpo_reader import HPOReader


class GMMConfig(Config):
	def __init__(self, d=None):
		super(GMMConfig, self).__init__()
		self.cov_type = 'full'
		self.n_init = 1
		self.max_iter = 20
		if d is not None:
			self.assign(d)


class GMMModel(SklearnModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR,
			mode=PREDICT_MODE, model_name=None, init_para=True):
		super(GMMModel, self).__init__(hpo_reader, vec_type, phe_list_mode)
		self.name = 'GMMModel' if model_name is None else model_name
		self.clf = None
		self.MODEL_SAVE_PATH = MODEL_PATH + os.sep + 'GMMModel' + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = MODEL_PATH + os.sep + 'GMMModel' + os.sep + self.name + '.json'
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)

		if init_para and mode == PREDICT_MODE:
			self.load()


	def predict_prob(self, X):
		if sp.issparse(X):
			X = X.toarray()
		return self.clf.predict_proba(X)


	@timer
	def train_X(self, X, y_, gmm_config, save_model=True):
		"""
		Args:
			X (np.ndarray): shape=(sample_num, feature_num)
			y_ (np.ndarray): shape=(sample_num,); -1 for unlabeled data
		"""
		print(gmm_config)
		print('X: ', X.shape, 'y_: ', y_.shape)
		self.clf = GaussianMixture(
			n_components=self.DIS_CODE_NUMBER, covariance_type=gmm_config.cov_type,
			max_iter=gmm_config.max_iter, n_init=gmm_config.n_init,
			means_init=np.array([X[y_==i].mean(axis=0) for i in range(self.DIS_CODE_NUMBER)]), verbose=2, verbose_interval=1
		)
		self.clf.fit(X, y_)
		if save_model:
			joblib.dump(self.clf, self.MODEL_SAVE_PATH)
			gmm_config.save(self.CONFIG_JSON)


def generate_model(vec_type, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		GMMModel
	"""
	model = GMMModel(hpo_reader, vec_type, phe_list_mode=phe_list_mode, model_name=model_name)
	if mode == PREDICT_MODE:
		model.load()
	return model


if __name__ == '__main__':
	pass





