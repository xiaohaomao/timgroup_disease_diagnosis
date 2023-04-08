import numpy as np
import joblib
import os
from sklearn.svm import LinearSVC

from core.predict.config import Config
from core.predict.model import SklearnModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PREDICT_MODE, VEC_COMBINE_MEAN, VEC_TYPE_0_1
from core.helper.data.data_helper import DataHelper


class LSVMConfig(Config):
	def __init__(self, d=None):
		super(LSVMConfig, self).__init__()
		self.C = 1.0
		self.max_iter = 3000
		if d is not None:
			self.assign(d)


class LSVMModel(SklearnModel):

	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, embed_mat=None,
			combine_modes=(VEC_COMBINE_MEAN,), dim_reductor=None, mode=PREDICT_MODE, model_name=None, save_folder=None,
			init_para=True, use_rd_mix_code=False):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
		"""
		super(LSVMModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, dim_reductor,
			use_rd_mix_code=use_rd_mix_code)
		self.name = 'LSVMModel' if model_name is None else model_name
		self.SAVE_FOLDER = save_folder
		self.clf = None
		if init_para and mode == PREDICT_MODE:
			self.load()


	def init_save_path(self):
		self.SAVE_FOLDER = self.SAVE_FOLDER or os.path.join(MODEL_PATH, self.hpo_reader.name, 'LSVMModel')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_SAVE_PATH = os.path.join(self.SAVE_FOLDER, self.name + '.joblib')
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, self.name + '.json')
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)



	def train(self, lsvm_config, save_model=True):
		# print(rf_config)
		raw_X, y_ = DataHelper(self.hpo_reader).get_train_raw_Xy(self.phe_list_mode, use_rd_mix_code=self.use_rd_mix_code)
		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, None, lsvm_config, save_model)


	def train_X(self, X, y_, sw, lsvm_config, save_model=True):
		self.clf = LinearSVC(
			C=lsvm_config.C, max_iter=lsvm_config.max_iter
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			self.save(self.clf, lsvm_config)


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		m = self.clf.decision_function(X)
		if len(self.clf.classes_) == 2:
			return np.vstack([1-m, m]).T
		return m


if __name__ =='__main__':
	pass

