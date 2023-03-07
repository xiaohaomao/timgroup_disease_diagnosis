

import shutil
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from memory_profiler import profile

from core.utils.constant import MODEL_PATH, PREDICT_MODE, TRAIN_MODE
from core.reader.hpo_filter_reader import HPOFilterReader
from core.predict.model import Model
from core.utils.utils import timer


class ClusterClassifyModel(Model):
	def __init__(self, hpo_reader, clt_generator, clf_generator, clt_kwargs={}, clf_kwargs={},
			mode=PREDICT_MODE, model_name=None, save_folder=None, fm_save_clt=False, init_para=True):
		super(ClusterClassifyModel, self).__init__()
		self.name = 'ClusterClassifyModel' if model_name is None else model_name
		self.hpo_reader = hpo_reader
		self.init_save_folder(save_folder)

		self.clt_generator = clt_generator
		self.clf_generator = clf_generator
		self.clt_kwargs = clt_kwargs
		self.clf_kwargs = clf_kwargs
		self.fm_save_clt = fm_save_clt

		self.clt = None
		self.lb_to_clf = None # {label: clf}
		self.lb_to_dis_code = None # {label: dis_code}

		if init_para and mode == PREDICT_MODE:
			self.load()


	def init_save_folder(self, save_folder):
		self.SAVE_FOLDER = save_folder or MODEL_PATH + os.sep + 'ClusterClassifyModel' + os.sep + self.name
		self.SAVE_CLT_FOLDER = self.SAVE_FOLDER + os.sep + 'clt'
		os.makedirs(self.SAVE_CLT_FOLDER, exist_ok=True)
		self.SAVE_CLF_FOLDER = self.SAVE_FOLDER + os.sep + 'clf'
		os.makedirs(self.SAVE_CLF_FOLDER, exist_ok=True)


	def get_clf_folder(self, lb):
		return self.SAVE_CLF_FOLDER + os.sep + str(lb)


	# @profile
	def train(self, clf_config, save_model=False, train_clf_cpu=1):
		self.train_clt(save_model=save_model)
		self.train_clf(self.clt, clf_config, save_model=save_model, cpu=train_clf_cpu)
		# self.clt.get_sim_model()


	@timer
	def train_clt(self, save_model=False):
		if self.fm_save_clt:
			self.clt = self.clt_generator(save_folder=None, **self.clt_kwargs)
		else:
			self.clt = self.clt_generator(save_folder=self.SAVE_CLT_FOLDER, **self.clt_kwargs)
		if self.clt.exists():
			print('Clt exists, load directly')
			self.clt.load()
		else:
			self.clt.train(save_model=save_model)


	# @timer
	# @profile
	def train_clf(self, clt, clf_config, save_model=False, cpu=1):
		self.lb_to_clf = {}
		self.init_lb_to_clf(clt)
		if cpu == 1 or cpu is None:
			for lb, clf in tqdm(self.lb_to_clf.items()):
				clf.train(clf_config, save_model=save_model)
		else:
			with Pool(cpu) as pool:
				print('cpu:', cpu)
				para_list = [(lb, clf, clf_config, save_model) for lb, clf in self.lb_to_clf.items()]
				chunksize = max(min(int(len(para_list) / cpu), 10), 1)
				for lb, clf in tqdm(pool.imap_unordered(self.train_clf_multi_wrap, para_list, chunksize=chunksize), total=len(para_list), leave=False):
					self.lb_to_clf[lb] = clf


	def train_clf_multi_wrap(self, paras):
		lb, clf, clf_config, save_model = paras
		clf.train(clf_config, save_model=save_model)
		return lb, clf


	def init_lb_to_clf(self, clt):
		print('Init clf list...')
		dis_list = self.hpo_reader.get_dis_list()
		self.lb_to_clf, self.lb_to_dis_code = {}, {}
		for lb, dis_int_list in tqdm(clt.get_LabelToDisIntList().items()):
			assert len(dis_int_list) != 0
			if len(dis_int_list) == 1:
				self.lb_to_dis_code[lb] = dis_list[dis_int_list[0]]
			else:
				hpo_reader = HPOFilterReader(keep_dis_int_set=set(dis_int_list))
				self.lb_to_clf[lb] = self.clf_generator(hpo_reader=hpo_reader, save_folder=self.get_clf_folder(lb), **self.clf_kwargs)


	def load(self):
		self.clt = self.clt_generator(**self.clt_kwargs)
		self.clt.load()
		self.init_lb_to_clf(self.clt)



	# @profile
	def query(self, phe_list, topk):
		"""
		Args:
			phe_list (list): list of phenotype
			topk (int or None): int--topk results with largest score (sorted by score); None--all result (sorted by score)
		Returns:
			list of tuple: [(dis1, score1), ...], scores decreasing
		"""
		ret_list = []
		lb_score_list = self.clt.predict(phe_list) # [(label, score), ...]
		for lb, lbscore in lb_score_list:
			if lb in self.lb_to_clf:
				dis_score_list = self.lb_to_clf[lb].query(phe_list, topk=None)
				ret_list.extend([(dis_code, (lbscore, score)) for dis_code, score in dis_score_list])
			else:
				ret_list.append((self.lb_to_dis_code[lb], (lbscore, 1.0)))
		assert len(ret_list) == self.hpo_reader.get_dis_num()
		if topk == None:
			return ret_list
		return ret_list[: topk]


	def query_many_multi_wrap(self, paras):
		return self.query(*paras)


	# @profile
	def query_many(self, phe_lists, topk=10, chunk_size=200, cpu_use=12):
		if cpu_use == 1:
			return [self.query(phe_list, topk) for phe_list in tqdm(phe_lists)]
		with Pool(cpu_use) as pool:
			para_list = [(phe_list, topk) for phe_list in phe_lists]
			return [result for result in tqdm(
				pool.imap(self.query_many_multi_wrap, para_list, chunksize=chunk_size),
				total=len(para_list), leave=False
			)]


	def delete_model(self):
		shutil.rmtree(self.SAVE_FOLDER, ignore_errors=True)


	def change_save_folder_and_save(self, model_name=None, save_folder=None):
		old_save_folder = self.SAVE_FOLDER
		self.name = model_name or self.name
		self.init_save_folder(save_folder)
		shutil.rmtree(self.SAVE_FOLDER, ignore_errors=True)
		os.system('cp -r {} {}'.format(old_save_folder, self.SAVE_FOLDER))


if __name__ == '__main__':
	pass



