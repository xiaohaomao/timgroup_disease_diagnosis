from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import os

from core.utils.constant import DATA_PATH, SIM_TO_DIST_MAX, SIM_TO_DIST_MIN_MAX_NORM, SIM_TO_DIST_MIN_MAX_SIGMOID, SIM_TO_DIST_E, SIM_TO_DIST_DIVIDE_PLUS_1
from core.utils.constant import PHELIST_REDUCE, PHELIST_ANCESTOR
from core.utils.constant import SET_SIM_SYMMAX, SET_SIM_ASYMMAX_QD, PHE_SIM_MICA, DIST_SHORTEST
from core.utils.constant import VEC_TYPE_0_1
from core.utils.constant import DIS_SIM_EUCLIDEAN, DIS_SIM_COSINE ,DIS_SIM_JACCARD, DIS_SIM_MICA, DIS_SIM_MICA_LIN, DIS_SIM_MIN_IC
from core.utils.constant import DIS_SIM_GDDP, DIS_SIM_ICTODQ_ACROSS, DIS_SIM_SIMTODQ_ACROSS, DIS_SIM_TREE_DISTANCE
from core.reader.hpo_reader import HPOReader
from core.predict.sim_model.mica_model import MICAModel
from core.predict.sim_model.mica_lin_model import MICALinModel
from core.predict.sim_model.sim_gic_model import SimGICModel
from core.predict.sim_model.min_ic_model import MinICModel
from core.predict.sim_model.gddp_fisher_model import GDDPFisherModel
from core.predict.sim_model.gddp_weight_term_overlap_model import GDDPWeightTOModel
from core.predict.sim_model.euclidean_model import EuclideanModel
from core.predict.sim_model.jaccard_model import JaccardModel
from core.predict.sim_model.cosine_model import CosineModel
from core.predict.sim_model.distance_model import DistanceModel
from core.predict.sim_model.sim_term_overlap_model import SimTOModel
from core.helper.data.data_helper import DataHelper
from core.utils.utils import min_max_norm, n_largest_indices, n_smallest_indices
from core.explainer.dis_sim_mat_explainer import DisSimMatExplainer


class DisSimCalculator(object):
	def __init__(self, hpo_reader=HPOReader(), dh=DataHelper()):
		self.SAVE_FOLDER = DATA_PATH+'/preprocess/model/DisSimCalculator'
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.hpo_reader = hpo_reader
		self.DIS_NUM = self.hpo_reader.get_dis_num()
		self.dh = dh

		self.sym_type_to_mat_func = {
			DIS_SIM_EUCLIDEAN: self.get_euclidean_mat, DIS_SIM_COSINE: self.get_cosine_mat, DIS_SIM_JACCARD: self.get_jaccard_mat,
			DIS_SIM_MICA: self.get_MICA_mat, DIS_SIM_MICA_LIN: self.get_MICA_lin_mat, DIS_SIM_MIN_IC : self.get_min_IC_mat,
			DIS_SIM_GDDP: self.get_gddp_fisher_mat, DIS_SIM_ICTODQ_ACROSS: self.get_ICTO_dq_accross_mat,
			DIS_SIM_SIMTODQ_ACROSS: self.get_sim_TO_q_reduce_mat, DIS_SIM_TREE_DISTANCE: self.get_tree_distance_mat
		}

		self.sym_type_to_model_gen_func = {
			DIS_SIM_EUCLIDEAN: EuclideanModel.generate_model, DIS_SIM_COSINE: CosineModel.generate_model,
			DIS_SIM_JACCARD: JaccardModel.generate_model, DIS_SIM_MICA: MICAModel.generate_model,
			DIS_SIM_MICA_LIN: MICALinModel.generate_model, DIS_SIM_MIN_IC: MinICModel.generate_model,
			DIS_SIM_GDDP: GDDPFisherModel.generate_model, DIS_SIM_ICTODQ_ACROSS: SimTOModel.generate_ICTO_dq_across_model,
			DIS_SIM_SIMTODQ_ACROSS: SimTOModel.generate_sim_TO_q_reduce_model, DIS_SIM_TREE_DISTANCE: DistanceModel.generate_model
		}


	def get_dis_sim_mat(self, dis_sim_type, positive=False, *args, **kwargs):
		sim_mat = self.sym_type_to_mat_func[dis_sim_type](*args, **kwargs)
		if positive:
			sim_mat = self.sim_mat_to_pos_sim_mat(sim_mat)
		return sim_mat


	def get_sim_model(self, dis_sim_type, *args, **kwargs):
		return self.sym_type_to_model_gen_func[dis_sim_type](*args, **kwargs)


	def sim_mat_to_pos_sim_mat(self, sim_mat):
		return sim_mat - min(np.min(sim_mat), 0)


	def sim_mat_to_dist_mat(self, sim_mat, change_way, max_Sim=None, dist_range=(0, 1)):
		assert np.isnan(sim_mat).any() == False    # no nan
		assert np.isinf(sim_mat).any() == False
		if change_way == SIM_TO_DIST_MAX:
			max_Sim = sim_mat.max() if max_Sim is None else max_Sim
			return max_Sim - sim_mat
		if change_way == SIM_TO_DIST_MIN_MAX_NORM:
			return dist_range[1] - min_max_norm(sim_mat, feature_range=dist_range)
		if change_way == SIM_TO_DIST_E:
			return np.exp(-sim_mat)
		if change_way == SIM_TO_DIST_MIN_MAX_SIGMOID:
			return 1 - 1 / (1 + np.exp(-sim_mat))
		if change_way == SIM_TO_DIST_DIVIDE_PLUS_1:
			return 1 / (1 + sim_mat)
		assert False


	def explain_all_default_mat(self):
		for sym_type, get_mat_func in self.sym_type_to_mat_func.items():
			print('-------------------------------------')
			print(sym_type)
			print(DisSimMatExplainer(get_mat_func(), sym_type, topk=20).explain())


	def is_symmetry_mat(self, m):
		return np.sum(m == m.T) == m.shape[0] * m.shape[1]


	def get_hpo_lists(self, phe_list_mode):
		dis_list = self.hpo_reader.get_dis_list()
		dis_to_hpo_list = self.hpo_reader.get_dis_to_hpo_dict(phe_list_mode)
		return [dis_to_hpo_list[dis_list[i]] for i in range(self.DIS_NUM)]


	def get_hpo_int_lists(self, phe_list_mode):
		dis_int_to_hpo_int = self.hpo_reader.get_dis_int_to_hpo_int(phe_list_mode)
		return [dis_int_to_hpo_int[i] for i in range(self.DIS_NUM)]


	def cal_euclidean_mat_multi_wrap(self, paras):
		model, i = paras
		return -model.cal_euclid_dist(model.dis_vec_mat[i]), i


	def get_euclidean_mat(self, phe_list_mode=PHELIST_ANCESTOR, vec_type=VEC_TYPE_0_1, cpu_use=12):
		folder = self.SAVE_FOLDER + '/Euclidean'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/m.npy'
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = EuclideanModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, vec_type=vec_type)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			para_list = [(model, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_euclidean_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		np.save(mat_Path, m)
		return m


	def get_cosine_mat(self, phe_list_mode=PHELIST_ANCESTOR):
		folder = self.SAVE_FOLDER + '/Cosine'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/m.npy'
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = CosineModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode)
		m = (model.dis_vec_mat * model.dis_vec_mat.T).A.astype(np.float32)
		np.save(mat_Path, m)
		return m


	def cal_jaccard_mat_multi_wrap(self, paras):
		model, i = paras
		return model.cal_score_for_phe_matrix(model.dis_vec_mat[i]), i


	def get_jaccard_mat(self, phe_list_mode=PHELIST_ANCESTOR, cpu_use=12):
		folder = self.SAVE_FOLDER + '/Jaccard'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/m.npy'
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = JaccardModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			para_list = [(model, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_jaccard_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		np.save(mat_Path, m)
		return m


	def cal_MICA_mat_multi_wrap(self, paras):
		model, hpo_int_lists, i = paras
		score_vec = np.array([model.phe_set_qd_sim(hpo_int_lists[i], hpo_int_lists[j]) for j in range(len(hpo_int_lists))])
		return score_vec, i


	def get_MICA_mat(self, phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX, cpu_use=12):
		folder = self.SAVE_FOLDER + '/MICA'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/{}-{}.npy'.format(phe_list_mode, set_sim_method)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = MICAModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, set_sim_method=set_sim_method)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			hpo_int_lists = self.get_hpo_int_lists(phe_list_mode)
			para_list = [(model, hpo_int_lists, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_MICA_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


	def cal_MICA_lin_mat_multi_wrap(self, paras):
		model, hpo_int_lists, i = paras
		score_vec = np.array([model.phe_set_qd_sim(hpo_int_lists[i], hpo_int_lists[j]) for j in range(len(hpo_int_lists))])
		return score_vec, i


	def get_MICA_lin_mat(self, phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX, cpu_use=12):
		folder = self.SAVE_FOLDER + '/MICALin'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/{}-{}.npy'.format(phe_list_mode, set_sim_method)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = MICALinModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, set_sim_method=set_sim_method)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			hpo_int_lists = self.get_hpo_int_lists(phe_list_mode)
			para_list = [(model, hpo_int_lists, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_MICA_lin_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


	def cal_min_IC_mat_multi_wrap(self, paras):
		model, hpo_int_lists, i = paras
		score_vec = np.array([model.phe_set_qd_sim(hpo_int_lists[i], hpo_int_lists[j]) for j in range(len(hpo_int_lists))])
		return score_vec, i


	def get_min_IC_mat(self, phe_list_mode=PHELIST_REDUCE, set_sim_method=SET_SIM_SYMMAX, cpu_use=12):
		folder = self.SAVE_FOLDER + '/MinIC'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/{}-{}.npy'.format(phe_list_mode, set_sim_method)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = MinICModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, set_sim_method=set_sim_method)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			hpo_int_lists = self.get_hpo_int_lists(phe_list_mode)
			para_list = [(model, hpo_int_lists, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_min_IC_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


	def cal_gddp_fisher_mat_multi_wrap(self, paras):
		model, hpo_list, i = paras
		score_vec = model.cal_score(hpo_list)
		return score_vec, i


	def get_gddp_fisher_mat(self, phe_list_mode=PHELIST_REDUCE, phe_sim=PHE_SIM_MICA, gamma=1.75, cpu_use=12):
		folder = self.SAVE_FOLDER + '/GDDPFisher'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/{}-{}-{}.npy'.format(phe_list_mode, phe_sim, gamma)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = GDDPFisherModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, phe_sim=phe_sim, gamma=gamma)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			hpo_lists = self.get_hpo_lists(phe_list_mode)
			para_list = [(model, hpo_lists[i], i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_gddp_fisher_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


	def get_ICTO_dq_accross_mat(self):
		folder = self.SAVE_FOLDER + '/ICTODQAccross'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/m.npy'
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = SimTOModel.generate_ICTO_dq_across_model(self.hpo_reader)
		hpo_lists = self.get_hpo_lists(PHELIST_REDUCE)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		for i in tqdm(range(self.DIS_NUM)):
			m[i] = model.cal_score(hpo_lists[i])
		assert self.is_symmetry_mat(m)
		np.save(mat_Path, m)
		return m


	def get_sim_TO_q_reduce_mat(self):
		folder = self.SAVE_FOLDER + '/SimTOQReduce'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/m.npy'
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = SimTOModel.generate_sim_TO_q_reduce_model(self.hpo_reader)
		hpo_lists = self.get_hpo_lists(PHELIST_REDUCE)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		for i in tqdm(range(self.DIS_NUM)):
			m[i] = model.cal_score(hpo_lists[i])
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


	def cal_tree_distance_mat_multi_wrap(self, paras):
		model, hpo_int_lists, i = paras
		score_vec = np.array([model.phe_set_qd_sim(hpo_int_lists[i], hpo_int_lists[j]) for j in range(len(hpo_int_lists))])
		return score_vec, i


	def get_tree_distance_mat(self, phe_list_mode=PHELIST_REDUCE, dist_type=DIST_SHORTEST, set_sim_method=SET_SIM_SYMMAX, cpu_use=12):
		folder = self.SAVE_FOLDER + '/TreeDistance'
		os.makedirs(folder, exist_ok=True)
		mat_Path = folder + '/{}-{}-{}.npy'.format(phe_list_mode, dist_type, set_sim_method)
		if os.path.exists(mat_Path):
			return np.load(mat_Path)
		model = DistanceModel.generate_model(self.hpo_reader, phe_list_mode=phe_list_mode, dist_type=dist_type, set_sim_method=set_sim_method)
		m = np.zeros(shape=(self.DIS_NUM, self.DIS_NUM), dtype=np.float32)
		with Pool(cpu_use) as pool:
			hpo_int_lists = self.get_hpo_int_lists(phe_list_mode)
			para_list = [(model, hpo_int_lists, i) for i in range(self.DIS_NUM)]
			for v, i in tqdm(pool.imap_unordered(self.cal_MICA_mat_multi_wrap, para_list, chunksize=200), total=self.DIS_NUM, leave=False):
				m[i] = v
		m = (m + m.T) / 2
		np.save(mat_Path, m)
		return m


if __name__ == '__main__':
	dsc = DisSimCalculator()
	m = dsc.sim_mat_to_dist_mat(dsc.get_jaccard_mat(), SIM_TO_DIST_MAX)
	explainer = DisSimMatExplainer(m, DIS_SIM_JACCARD+'-'+SIM_TO_DIST_MAX, topk=100000)
	explainer.explain_and_save(); explainer.draw_dist()

