from core.predict.model import SklearnModel
from core.predict.config import Config
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PREDICT_MODE
from core.utils.constant import PHELIST_ANCESTOR
import time
import joblib
import os
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeConfig(Config):
	def __init__(self):
		super(DecisionTreeConfig, self).__init__()
		self.max_leaf_nodes=None



class DecisionTreeModel(SklearnModel):
	def __init__(self, hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor=None, model_name=None):
		"""
		vec_type (str): VEC_TYPE_0_1 |
		"""
		super(DecisionTreeModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_path, dim_reductor)
		self.name = 'DecisionTreeModel_01_Ances' if model_name is None else model_name

		self.clf = None
		self.MODEL_SAVE_PATH = MODEL_PATH + os.sep + 'DecisionTreeModel' + os.sep + self.name + '.joblib'
		self.CONFIG_JSON = MODEL_PATH + os.sep + 'DecisionTreeModel' + os.sep + self.name + '.json'
		os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)


	def train(self, raw_X, y_, sw, dt_config, logger):

		logger.info('training %s' % (self.name,))
		logger.info(dt_config)
		X = self.raw_X_to_X_func(raw_X)
		self.clf = DecisionTreeClassifier(max_leaf_nodes=dt_config.max_leaf_nodes)

		begin_time = time.time(); logger.info('Training begin..')
		self.clf.fit(X, y_, sample_weight=sw)
		logger.info('Training end. Elapsed time: {:.1f} minutes'.format((time.time() - begin_time)/60))

		joblib.dump(self.clf, self.MODEL_SAVE_PATH)
		dt_config.save(self.CONFIG_JSON)


def generate_model(vec_type, hpo_reader=HPOReader(), phe_list_mode=PHELIST_ANCESTOR, embed_path=None, mode=PREDICT_MODE, model_name=None):
	"""
	Returns:
		DecisionTreeModel
	"""
	model = DecisionTreeModel(hpo_reader, vec_type, phe_list_mode=phe_list_mode, embed_path=embed_path, model_name=model_name)
	if mode == PREDICT_MODE:
		model.load()
	return model


if __name__ == '__main__':
	pass