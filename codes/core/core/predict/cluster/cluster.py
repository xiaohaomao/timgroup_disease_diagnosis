from copy import deepcopy
import os
import numpy as np
import heapq
from collections import Counter

from core.predict.config import Config
from core.reader.hpo_reader import HPOReader
from core.predict.calculator.dis_sim_calculator import DisSimCalculator
from core.utils.constant import DIS_SIM_MICA, PREDICT_MODE, TRAIN_MODE, CLUSTER_PREDICT_MEAN_MAX_TOPK, CLUSTER_PREDICT_MEAN, CLUSTER_PREDICT_CENTER
from core.utils.utils import check_return
from core.explainer.cluster_explainer import ClusterExplainer
from core.helper.hyper.file_manager import FileManager


class ClusterConfig(Config):
	def __init__(self):
		super(ClusterConfig, self).__init__()
		self.dis_sim_type = DIS_SIM_MICA
		self.sim_kwargs = {}
		self.predict_method = CLUSTER_PREDICT_MEAN_MAX_TOPK
		self.topk = 20


class Cluster(object):
	def __init__(self, c, hpo_reader=HPOReader(), mode=PREDICT_MODE, save_folder=None):
		self.name = 'Cluster'
		self.c = c
		self.hpo_reader = hpo_reader
		self.mode = mode
		self.SAVE_FOLDER = save_folder
		self.dis_list = hpo_reader.get_dis_list()

		self.labels = None
		self.unq_labels = None
		self.sim_model = None
		self.label_to_size = None
		self.label_to_dis_int_list = None

		self.LABEL_SAVE_NPY = None
		self.CONFIG_SAVE_JSON = None

		self.have_init_save_path = False


	def train(self, c):
		raise NotImplementedError


	def get_default_fm_folder(self):
		raise NotImplementedError


	def get_label_to_center(self, clt=None):
		"""
		Returns:
			dict: {label: dis_int}
		"""
		raise NotImplementedError


	@check_return('labels')
	def get_labels(self, clt=None):
		"""
		Returns:
			np.ndarray: shape=(sample_num,)
		"""
		return clt.labels_


	@check_return('sim_model')
	def get_sim_model(self):
		return DisSimCalculator(self.hpo_reader).get_sim_model(self.c.dis_sim_type, **self.c.sim_kwargs)


	@check_return('label_to_size')
	def get_label_to_size(self):
		return Counter(self.get_labels())


	@check_return('unq_labels')
	def get_unique_labels(self):
		"""
		Returns:
			list: [label1, label2]
		"""
		return list(self.get_label_to_size().keys())


	@check_return('label_to_dis_int_list')
	def get_LabelToDisIntList(self):
		labels = self.get_labels()
		unq_labels = self.get_unique_labels()
		d = {lb: [] for lb in unq_labels}
		for i, lb in enumerate(labels):
			d[lb].append(i)
		return d


	def get_cluster_num(self):
		return len(self.get_unique_labels())


	def sim_mat_to_dist_mat(self, sim_mat):
		# FIXME: np.diag(sim_mat) = 0?
		return 1.0 - (sim_mat / np.diag(sim_mat))


	def init_save_path(self, create=False):
		if self.have_init_save_path:
			return
		self.SAVE_FOLDER = self.SAVE_FOLDER or self.get_default_save_folder(create)
		assert self.SAVE_FOLDER is not None
		if create:
			os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.LABEL_SAVE_NPY = self.SAVE_FOLDER + '/labels.npy'
		self.CONFIG_SAVE_JSON = self.SAVE_FOLDER + '/config.json'
		self.have_init_save_path = True


	def get_file_manager(self):
		return FileManager(self.get_default_fm_folder())


	def get_default_save_folder(self, create=False):
		fm = self.get_file_manager()
		id = self.get_id()
		return fm.add_id(id) if create else fm.get_path(id)


	def save(self):
		self.init_save_path(create=True)
		np.save(self.LABEL_SAVE_NPY, self.get_labels())
		self.c.save(self.CONFIG_SAVE_JSON)
		explainer = ClusterExplainer(self)
		explainer.explain_save_json(self.SAVE_FOLDER+'/explain.json')
		explainer.draw(self.SAVE_FOLDER)


	def load(self):
		self.init_save_path(create=False)
		self.labels = np.load(self.LABEL_SAVE_NPY)


	def config_to_id_dict(self):
		d = deepcopy(self.c.__dict__)
		del d['topk'], d['predict_method']
		return d


	def get_id(self):
		return str(self.config_to_id_dict())


	def exists(self):
		save_folder = self.SAVE_FOLDER or self.get_default_save_folder(create=False)
		if save_folder is None:
			return False
		return os.path.exists(save_folder + '/labels.npy')


	def cal_score(self, phe_list):
		"""
		Args:
			phe_list (list): [hpo_code1, hpo_code2, ...]
		Returns:
			np.ndarray: score_vec, shape=[dis_num]
		"""
		return self.get_sim_model().cal_score(phe_list)


	def predict(self, phe_list):
		if self.c.predict_method == CLUSTER_PREDICT_MEAN_MAX_TOPK:
			return self.predict_mean_max_topk(phe_list, self.c.topk)
		if self.c.predict_method == CLUSTER_PREDICT_MEAN:
			return self.predict_mean(phe_list)
		if self.c.predict_method == CLUSTER_PREDICT_CENTER:
			return self.predict_center(phe_list)
		assert False


	def predict_mean_max_topk(self, phe_list, topk):
		"""
		Args:
			phe_list (list): [hpo_code, ...]
		Returns:
			list: [(label, score), ...]; score from big -> small
		"""
		score_vec = self.cal_score(phe_list)
		lb_to_dis_int = self.get_LabelToDisIntList()
		ret_list = []
		for lb, size in self.get_label_to_size().items():
			score = np.median(heapq.nlargest(min(topk, size), score_vec[lb_to_dis_int[lb]]))
			ret_list.append((lb, score))
		return sorted(ret_list, key=lambda item: item[1], reverse=True)


	def predict_mean(self, phe_list):
		return self.predict_mean_max_topk(phe_list, topk=np.inf)


	def predict_center(self, phe_list):
		score_vec = self.cal_score(phe_list)
		lb_to_cdis_int = self.get_label_to_center()
		return sorted([(lb, score_vec[cDisInt]) for lb, cDisInt in lb_to_cdis_int.items()], key=lambda item: item[1], reverse=True)


if __name__ == '__main__':
	pass




