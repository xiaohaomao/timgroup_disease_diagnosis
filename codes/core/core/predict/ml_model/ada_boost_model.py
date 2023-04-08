from core.predict.config import Config
from core.predict.model import SklearnModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PREDICT_MODE
import joblib
import os
from sklearn.ensemble import AdaBoostClassifier


class AdaConfig(Config):
	def __init__(self):
		super(AdaConfig, self).__init__()
		self.base_estimator = None
		self.n_estimators = 50
		self.lr = 1.0
		self.algorithm = 'SAMME.R'


class AdaBoostModel(SklearnModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode, embed_path=None, dim_reductor=None, model_name=None):
		super(AdaBoostModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor)
		self.name = 'AdaBoostModel' if model_name is None else model_name

		self.clf = None
		self.MODEL_SAVE_PATH = MODEL_PATH + os.sep + 'AdaBoostModel' + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = MODEL_PATH + os.sep + 'AdaBoostModel' + os.sep + self.name + '.json'
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)


	def train(self, raw_X, y_, sw, ada_config, logger, save_model=True):

		print(ada_config)

		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, sw, ada_config, save_model)


	def train_X(self, X, y_, sw, ada_config, save_model=True):
		self.clf = AdaBoostClassifier(
			base_estimator=ada_config.base_estimator, n_estimators=ada_config.n_estimators,
			learning_rate=ada_config.lr, algorithm=ada_config.algorithm
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			joblib.dump(self.clf, self.MODEL_SAVE_PATH)
			ada_config.save(self.CONFIG_JSON)


def generate_model(vec_type, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, embed_path=None, dim_reductor=None, mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		AdaBoostModel
	"""
	model = AdaBoostModel(hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor, model_name)
	if mode == PREDICT_MODE:
		model.load()
	return model


if __name__ =='__main__':
	pass