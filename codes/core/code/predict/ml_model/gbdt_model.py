

from core.predict.config import Config
from core.predict.config import Config
from core.predict.model import SklearnModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PREDICT_MODE, VEC_COMBINE_MEAN
from core.utils.utils import data_to_01_matrix
import lightgbm as lgb
import joblib
import os
import numpy as np


class GBDTConfig(Config):
	def __init__(self):
		super(GBDTConfig, self).__init__()
		self.tree_num = 100
		self.max_leaves = 31
		self.lr = 0.1
		self.bagging_frac = 1.0
		self.bagging_freq = 5
		self.feature_frac = 1.0
		self.l2_lambda = 0.0
		self.n_jobs = 12


class GBDTModel(SklearnModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode, embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,), dim_reductor=None, model_name=None):
		super(GBDTModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, dim_reductor)
		self.name = 'GBDTModel' if model_name is None else model_name

		self.clf = None
		self.MODEL_SAVE_PATH = MODEL_PATH + os.sep + 'GBDTModel' + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = MODEL_PATH + os.sep + 'GBDTModel' + os.sep + self.name + '.json'
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)


	def raw_X_to_01_X(self, raw_X):
		return data_to_01_matrix(raw_X, self.HPO_CODE_NUMBER, dtype=np.float32)


	def train(self, raw_X, y_, sw, gbdt_config, logger, save_model=True):

		logger.info(self.name)
		logger.info(gbdt_config)

		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, sw, gbdt_config, save_model)


	def train_X(self, X, y_, sw, gbdt_config, save_model=True):
		self.clf = lgb.LGBMClassifier(
			boosting_type='gbdt', num_leaves=gbdt_config.max_leaves, learning_rate=gbdt_config.lr, n_estimators=gbdt_config.tree_num,
			subsample=gbdt_config.bagging_frac, subsample_freq=gbdt_config.bagging_freq, colsample_bytree=gbdt_config.feature_frac,
			reg_lambda=gbdt_config.l2_lambda, n_jobs=gbdt_config.n_jobs
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			joblib.dump(self.clf, self.MODEL_SAVE_PATH)
			gbdt_config.save(self.CONFIG_JSON)


	def delete(self):
		os.remove(self.MODEL_SAVE_PATH)
		os.remove(self.CONFIG_JSON)


def generate_model(vec_type, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, embed_mat=None, combine_modes=(VEC_COMBINE_MEAN,),
				dim_reductor=None, mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		GBDTModel
	"""
	model = GBDTModel(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes, dim_reductor, model_name)
	if mode == PREDICT_MODE:
		model.load()
	return model


if __name__ =='__main__':
	pass
