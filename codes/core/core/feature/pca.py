
from core.predict.config import Config
from core.utils.constant import MODEL_PATH
from core.utils.utils import list_to_str_with_step
from sklearn.decomposition import PCA
import joblib
from sklearn.utils import check_array
import os

class PCAConfig(Config):
	def __init__(self):
		super(PCAConfig, self).__init__()
		self.n_component = 'mle' # n_components='mle' is only supported if n_samples >= n_features...
		self.svd_solver = 'auto'


class PCADimReductor(object):
	def __init__(self, name=None):
		self.name = 'PCADimReductor' if name is None else name
		folder = MODEL_PATH + os.sep + 'PCADimReductor' + os.sep + self.name; os.makedirs(folder, exist_ok=True)
		self.SAVE_MODEL_PATH = folder+os.sep+'pca.m'
		self.SAVE_CONFIG_PATH = folder+os.sep+'config.json'
		self.SAVE_LOG_PATH = folder+os.sep+'log'
		self.pca = None


	def load(self):
		self.pca = joblib.load(self.SAVE_MODEL_PATH)


	def save(self):
		joblib.dump(self.pca, self.SAVE_MODEL_PATH)


	def train(self, X, pca_config):
		"""
		Args:
			X (np.ndarray): shape=(sample_num, feature_num)
		Returns:
			np.ndarray: new X, shape=(sample_num, nConponents)
		"""
		if pca_config.n_component == 'mle':
			pca_config.svd_solver = 'full'
		self.pca = PCA(n_components=pca_config.n_component, svd_solver=pca_config.svd_solver)
		new_X = self.pca.fit_transform(X)
		self.save()
		self.output_train_log(X.shape[1])
		return new_X


	def output_train_log(self, feature_num):
		import numpy as np
		step = 6
		var_accumulate_ratio = np.copy(self.pca.explained_variance_ratio_)
		for i in range(1, len(self.pca.explained_variance_ratio_)):
			var_accumulate_ratio[i] += var_accumulate_ratio[i-1]
		s = 'Components Number = {} (Original Feature Number = {})\n' \
			'Variance: \n{}\n' \
			'Variance Ratio: \n{}\n' \
			'Variance Accumulated Ratio: \n{}\n'.format(
			self.pca.n_components_, feature_num,
			list_to_str_with_step(self.pca.explained_variance_, step),
			list_to_str_with_step(self.pca.explained_variance_ratio_, step),
			list_to_str_with_step(var_accumulate_ratio, step),
		)
		print(s, file=open(self.SAVE_LOG_PATH, 'w'))


	def transform(self, X):
		"""
		Args:
			X (np.ndarray): shape=(sample_num, feature_num)
		Returns:
			np.ndarray: new X, shape=(sample_num, nConponents)
		"""
		return self.pca.transform(X)


if __name__ == '__main__':
	pass






