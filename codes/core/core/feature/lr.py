

from core.reader.hpo_reader import HPOReader
from core.utils.utils import data_to_01_matrix, timer, get_logger, getDisHPOMat
from core.script.showFeature import show_feature_weight
from core.utils.constant import PHELIST_ANCESTOR, MODEL_PATH, RESULT_PATH
from core.predict.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import json

class LRConfig(Config):
	def __init__(self):
		super(LRConfig, self).__init__()
		self.penalty = 'l2'
		self.C = 1.0    #
		self.fit_intercept = True


class LRFeatureSelector(object):
	def __init__(self, hpo_reader, name=None):
		self.name = 'LRFeatureSelector' if name is None else name
		self.hpo_reader = hpo_reader

		self.w = None
		self.b = None
		self.row_names, self.col_names = None, None
		self.folder = MODEL_PATH + os.sep + 'LRFeatureSelector' + os.sep + self.name; os.makedirs(self.folder, exist_ok=True)
		self.w_npz = self.folder + os.sep + 'w.npz'
		self.config_json = self.folder + os.sep + 'config.json'


	def get_W(self):
		"""get feature weights matrix
		Returns:
			np.ndarray: w, shape=(dis_num, feature_num)
			np.ndarray: row_names, shape=(w.shape[0],)
			np.ndarray: col_names, shape=(w.shape[1],)
		"""
		if self.w is None:
			self.load()
		return self.w, self.row_names, self.col_names


	@timer
	def save(self):
		self.c.save(self.config_json)
		np.savez_compressed(self.w_npz, w=self.w, b=self.b, row_names=self.row_names, col_names=self.col_names)


	@timer
	def load(self):
		data = np.load(self.w_npz)
		self.w, self.b, self.row_names, self.col_names = data['w'], data['b'], data['row_names'], data['col_names']


	def get_X_and_dim_names(self):
		return getDisHPOMat(self.hpo_reader, PHELIST_ANCESTOR)


	def train_single(self, paras):


		X, row_id = paras
		y_ = np.zeros(shape=(X.shape[0], ), dtype=np.int32); y_[row_id] = 1

		clf = LogisticRegression(
			penalty=self.c.penalty, C = self.c.C, fit_intercept=self.c.fit_intercept
		)
		clf.fit(X, y_)

		y = clf.predict(X)
		y_score = clf.predict_proba(X)[:, 1].flatten()
		sw = compute_sample_weight('balanced', y_); sw=None
		acc = accuracy_score(y_, y, sample_weight=sw)
		auc = roc_auc_score(y_, y_score, sample_weight=sw)
		return clf, row_id, acc, auc


	def train(self, lr_config):
		self.c = lr_config
		self.init_train_log()
		X, self.row_names, self.col_names = self.get_X_and_dim_names()
		w = np.zeros(shape=X.shape, dtype=np.float32)
		b = np.zeros(shape=X.shape[0], dtype=np.float32)
		paras = [(X, dis_int) for dis_int in range(X.shape[0])]
		with Pool() as pool:
			for clf, row_id, acc, auc in tqdm(pool.imap_unordered(self.train_single, paras), total=len(paras), leave=False):
				self.push_train_log(clf, row_id, acc, auc)
				w[row_id, :] = clf.coef_.flatten()
				b[row_id] = clf.intercept_
		self.w = w; self.b = b
		self.save()
		self.output_train_log(self.folder + os.sep + 'log')


	def init_train_log(self):
		self.log_bucket = []


	def push_train_log(self, clf, row_id, acc, auc):
		self.log_bucket.append((row_id, acc, auc))
		del clf


	def output_train_log(self, filepath):
		from core.utils.utils import addDisInfo, add_info
		row_id_to_info = {i: info for i, info in enumerate(addDisInfo(self.hpo_reader.get_dis_list()))}
		bucket_size = len(self.log_bucket)
		row_ids, accs, aucs = zip(*sorted(self.log_bucket))
		row_id_infos = add_info(row_ids, row_id_to_info, lambda tgt: isinstance(tgt, int))
		s = ''; k = 10
		s += 'Average Train Accuracy={ave_acc}; Average Train AUC={ave_auc}\n'.format(ave_acc=np.mean(accs), ave_auc=np.mean(aucs))
		order_acc = sorted(zip(accs, row_id_infos), reverse=True)
		s += 'topk Acc: {topk_acc}\n'.format(topk_acc=order_acc[:k])
		s += 'lastk Acc: {lastk_acc}\n'.format(lastk_acc=order_acc[-k:])
		order_auc = sorted(zip(aucs, row_id_infos), reverse=True)
		s += 'topk AUC: {topk_auc}\n'.format(topk_auc=order_auc[:k])
		s += 'lastk AUC: {lastk_auc}\n'.format(lastk_auc=order_auc[-k:])
		s += '\n=============================================================\n\n'
		for i in range(bucket_size):
			s += '{row_id} {dis}: acc={acc}; auc={auc}\n'.format(row_id=row_ids[i], dis=row_id_to_info[row_ids[i]], acc=accs[i], auc=aucs[i])
		print(s, file=open(filepath, 'w'))


if __name__ == '__main__':
	folder = RESULT_PATH+os.sep+'FeatureWeight'+os.sep+'LRFeatureSelector'; os.makedirs(folder, exist_ok=True)
	def train_script():
		CList = [0.001]
		for C in CList:
			hpo_reader = HPOReader()
			lr_config = LRConfig()
			lr_config.C = C
			selector_name = 'LRFeatureSelector_NoBla_C{C}'.format(C=C)
			fs = LRFeatureSelector(hpo_reader, name=selector_name)
			fs.train(lr_config)
			X, _, _ = fs.get_X_and_dim_names()
			show_feature_weight(fs.w, fs.b, fs.row_names, fs.col_names, X, folder+os.sep+selector_name, k=10)

	train_script()