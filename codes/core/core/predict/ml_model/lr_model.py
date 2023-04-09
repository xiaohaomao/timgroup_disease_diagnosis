import os
import joblib
from sklearn.linear_model import LogisticRegression
from core.predict.model import SklearnModel
from core.predict.config import Config
from core.reader.hpo_reader import HPOReader
from core.utils.constant import MODEL_PATH, PREDICT_MODE, VEC_TYPE_0_1
from core.utils.constant import PHELIST_ANCESTOR, VEC_COMBINE_MEAN
from core.utils.utils import timer, item_list_to_rank_list
from core.helper.data.data_helper import DataHelper

class LogisticConfig(Config):
	def __init__(self, d=None):
		super(LogisticConfig, self).__init__()
		self.C = 0.008
		self.penalty = 'l2'
		self.solver = 'liblinear'
		self.max_iter = 300
		self.n_jobs = 12
		self.multi_class = 'ovr'
		self.class_weight = 'balanced'
		self.fit_intercept = True
		if d is not None:
			self.assign(d)


class LogisticModel(SklearnModel):
	def __init__(self, hpo_reader=HPOReader(), vec_type=VEC_TYPE_0_1, phe_list_mode=PHELIST_ANCESTOR, embed_mat=None,
			combine_modes=(VEC_COMBINE_MEAN,), dim_reductor=None, mode=PREDICT_MODE, model_name=None, save_folder=None,
			init_para=True, use_rd_mix_code=False):
		"""
		Args:
			hpo_reader (HPOReader or HPOFilterDatasetReader)
			vec_type (str): VEC_TYPE_0_1 | VEC_TYPE_0_1_DIM_REDUCT | VEC_TYPE_EMBEDDING
		"""
		super(LogisticModel, self).__init__(hpo_reader, vec_type, phe_list_mode, embed_mat, combine_modes,
			dim_reductor, use_rd_mix_code=use_rd_mix_code)
		self.name = 'LogisticModel' if model_name is None else model_name
		self.SAVE_FOLDER = save_folder
		self.clf = None
		if init_para and mode == PREDICT_MODE:
			self.load()


	def init_save_path(self):
		self.SAVE_FOLDER = self.SAVE_FOLDER or os.path.join(MODEL_PATH, self.hpo_reader.name, 'LogisticModel')
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_SAVE_PATH = os.path.join(self.SAVE_FOLDER, '{}.joblib'.format(self.name))
		self.CONFIG_JSON = os.path.join(self.SAVE_FOLDER, '{}.json'.format(self.name))

	def train(self, lr_config, save_model=True):
		raw_X, y_ = DataHelper(self.hpo_reader).get_train_raw_Xy(self.phe_list_mode, use_rd_mix_code=self.use_rd_mix_code)
		X = self.raw_X_to_X_func(raw_X)
		self.train_X(X, y_, None, lr_config, save_model)


	def train_X(self, X, y_, sw, lr_config, save_model=True):
		self.clf = LogisticRegression(
			C=lr_config.C, penalty=lr_config.penalty, solver=lr_config.solver, max_iter=lr_config.max_iter, n_jobs=lr_config.n_jobs,
			multi_class=lr_config.multi_class, class_weight=lr_config.class_weight, fit_intercept=lr_config.fit_intercept
		)
		self.clf.fit(X, y_, sample_weight=sw)
		if save_model:
			self.save(self.clf, lr_config)


	def explain(self, pa_hpo_list, dis_code):
		"""return HPO importance that illustrate why the patient is diagnosed as dis_code
		Returns:
			list: [(hpo_code, importance), ...]; ordered by importance, from big to small
		"""
		row_idx = self.hpo_reader.get_dis_map_rank()[dis_code]
		hpo_map_rank = self.hpo_reader.get_hpo_map_rank()
		return sorted([(hpo_code, self.clf.coef_[row_idx][hpo_map_rank[hpo_code]]) for hpo_code in pa_hpo_list], key=lambda item: item[1], reverse=True)


	def explain_as_str(self, pa_hpo_list, dis_code):
		from core.explainer.explainer import Explainer
		explain_list = Explainer().add_cns_info(self.explain(pa_hpo_list, dis_code))
		return '\n'.join([str(item) for item in explain_list])


	def predict_prob(self, X):
		"""
		Args:
			X (array-like or sparse matrix): shape=(sample_num, feature_num)
		Returns:
			np.array: shape=[sample_num, class_num]
		"""
		return self.clf.predict_log_proba(X)


if __name__ == '__main__':
	pass




