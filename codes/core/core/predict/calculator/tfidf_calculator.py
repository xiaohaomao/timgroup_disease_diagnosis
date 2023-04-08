import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from core.utils.utils import check_load_save, SPARSE_NPZ_FILE_FORMAT
from core.utils.constant import DATA_PATH, JOBLIB_FILE_FORMAT, PHELIST_ANCESTOR_DUP, VEC_TYPE_TF
import core.helper.data.data_helper as dhelper


class TFIDFCalculator(object):
	def __init__(self):
		self.OUTPUT_FOLDER = DATA_PATH + '/preprocess/model/TFIDFCalculator'
		os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

		self.DEFAULT_TRANSFORMER_PATH = self.OUTPUT_FOLDER + '/DefaultTransformer.pkl'
		self.default_transformer = None
		self.DEFAULT_DIS_HPO_MAT_NPZ = self.OUTPUT_FOLDER + '/DefaultDisHPOMat.npz'
		self.default_dis_hpo_mat = None


	@check_load_save('default_transformer', 'DEFAULT_TRANSFORMER_PATH', JOBLIB_FILE_FORMAT)
	def get_default_transformer(self):
		"""ref: # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
		"""
		dis_hpo_tf_mat = self.get_dis_hpo_tf_mat()
		transformer = TfidfTransformer()
		transformer.fit(dis_hpo_tf_mat)
		return transformer


	@check_load_save('default_dis_hpo_mat', 'DEFAULT_DIS_HPO_MAT_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_default_dis_hpo_mat(self):
		"""
		Returns:
			scipy.sparse.csr.csr_matrix: shape=(dis_num, hpo_num)
		"""
		dis_hpo_tf_mat = self.get_dis_hpo_tf_mat()
		return self.get_default_transformer().transform(dis_hpo_tf_mat)


	def get_dis_hpo_tf_mat(self):
		"""
		Returns:
			scipy.sparse.csr.csr_matrix: shape=(dis_num, hpo_num)
		"""
		return dhelper.DataHelper().get_train_X(PHELIST_ANCESTOR_DUP, VEC_TYPE_TF, sparse=True, dtype=np.int32)


if __name__ == '__main__':
	calculator = TFIDFCalculator()
	calculator.get_default_transformer()
	calculator.get_default_dis_hpo_mat()



