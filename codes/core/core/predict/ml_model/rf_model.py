from core.predict.config import Config
from core.predict.model import SklearnModel
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PHELIST_ANCESTOR, PREDICT_MODE
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

class RFConfig(Config):
	def __init__(self):
		super(RFConfig, self).__init__()
		self.n_estimators = 10
		self.criterion = 'gini'
		self.max_leaf_nodes = None
		self.max_features = 'auto'
		self.n_jobs = 16


class RFModel(SklearnModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode, embed_path=None, dim_reductor=None, model_name=None):
		super(RFModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor)
		self.name = 'RFModel' if model_name is None else model_name

		self.clf = None
		self.MODEL_SAVE_PATH = MODEL_PATH + os.sep + 'RFModel' + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = MODEL_PATH + os.sep + 'RFModel' + os.sep + self.name + '.json'
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)


	def train(self, raw_X, y_, sw, rf_config, logger, save_model=True):

		print(rf_config)

		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, sw, rf_config, save_model)


	def train_X(self, X, y_, sw, rf_config, save_model=True):
		self.clf = RandomForestClassifier(
			n_estimators=rf_config.n_estimators, criterion=rf_config.criterion, n_jobs=rf_config.n_jobs,
			max_leaf_nodes=rf_config.max_leaf_nodes, max_features=rf_config.max_features
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			joblib.dump(self.clf, self.MODEL_SAVE_PATH)
			rf_config.save(self.CONFIG_JSON)


def generate_model(vec_type, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, embed_path=None, dim_reductor=None, mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		RFModel
	"""
	model = RFModel(hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor, model_name)
	if mode == PREDICT_MODE:
		model.load()
	return model


if __name__ =='__main__':
	pass










