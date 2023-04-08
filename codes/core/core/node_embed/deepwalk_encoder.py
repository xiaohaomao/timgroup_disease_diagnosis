from core.node_embed.encoder import Encoder
from core.predict.config import Config
from core.node_embed.deepwalk import graph
from core.node_embed.deepwalk.skipgram import Skipgram
from core.node_embed.deepwalk import walks as serialized_walks
from core.utils.constant import GRAPH_ADJLIST, GRAPH_EDGELIST, DATA_PATH, EMBEDDING_PATH
from core.utils.utils import gen_adj_list, get_edge_list, get_logger, timer
import random
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import os


class DeepwalkConfig(Config):
	def __init__(self):
		super(DeepwalkConfig, self).__init__()
		self.num_walks = 30
		self.walkLength = 40
		self.seed = 0
		self.window_size = 5
		self.workers = int(os.cpu_count() / 2)
		self.embed_size = 128
		self.undirected = True
		self.graph_format = GRAPH_ADJLIST
		self.maxMem = 1000000000    # 1G
		self.vertexFreqDegree = False
		self.skipGram = 1
		self.hierarchySoftmax = 1   #
		self.negative = 20
		self.iterNum = 5


class DeepwalkEncoder(Encoder):
	def __init__(self, encoder_name=None):
		super(DeepwalkEncoder, self).__init__()
		self.name = 'DeepwalkEncoder' if encoder_name is None else encoder_name
		folder = EMBEDDING_PATH + os.sep + 'DeepwalkEncoder'; os.makedirs(folder, exist_ok=True)
		self.EMBED_PATH = folder + os.sep + self.name + '.npz'
		self.CONFIG_JSON = folder + os.sep + self.name + '.json'
		self.LOG_PATH = folder + os.sep + self.name + '.log'
		self.WALKS_BASE_PATH = folder + os.sep + self.name + '.walks'
		self.hpo_embed = None


	def get_embed(self):
		"""
		Returns:
			np.ndarray: shape=[hpo_num, vec_size]
		"""
		if self.hpo_embed is None:
			with np.load(self.EMBED_PATH) as data:
				self.hpo_embed = data['arr_0']
		return self.hpo_embed


	def get_graph_from_adj(self, undirected, hpo_reader):
		GRAPH_PATH = DATA_PATH + os.sep + 'preprocess' + os.sep + 'HPO_GRAPH.adjlist'
		if not os.path.exists(GRAPH_PATH):
			gen_adj_list(hpo_reader.get_hpo_dict(), hpo_reader.get_hpo_map_rank(), GRAPH_PATH)
		return graph.load_adjacencylist(GRAPH_PATH, undirected=undirected)


	def get_graph_from_edge(self, undirected, hpo_reader):
		GRAPH_PATH = DATA_PATH + os.sep + 'preprocess' + os.sep + 'HPO_GRAPH.edgelist'
		if not os.path.exists(GRAPH_PATH):
			get_edge_list(hpo_reader.get_hpo_dict(), hpo_reader.get_hpo_map_rank(), GRAPH_PATH)
		return graph.load_edgelist(GRAPH_PATH, undirected=undirected)


	def get_graph(self, undirected, hpo_reader, graph_format=GRAPH_ADJLIST):
		if graph_format == GRAPH_ADJLIST:
			return self.get_graph_from_adj(undirected, hpo_reader)
		elif graph_format == GRAPH_EDGELIST:
			return self.get_graph_from_edge(undirected, hpo_reader)
		else:
			raise Exception("Unknown file format: '%s'.	Valid formats: 'adjlist', 'edgelist', 'mat'" % graph_format.format)


	@timer
	def train(self, c, hpo_reader):
		logger = get_logger(self.name, log_path=self.LOG_PATH, mode='w')
		logger.info(self.name)
		logger.info(c)

		G = self.get_graph(c.undirected, hpo_reader)
		logger.info("Number of nodes: {}".format(len(G.nodes())))

		num_walks = len(G.nodes()) * c.num_walks
		logger.info("Number of walks: {}".format(num_walks))

		data_size = num_walks * c.walkLength
		logger.info("Data size (walks*length): {}".format(data_size))

		if data_size < c.maxMem:
			logger.info("Walking...")
			walks = graph.build_deepwalk_corpus(
				G, num_paths=c.num_walks, path_length=c.walkLength, alpha=0, rand=random.Random(c.seed)
			)
			logger.info("Training...")
			model = Word2Vec(
				walks, size=c.embed_size, window=c.window_size, min_count=0,
				sg=c.skipGram, hs=c.hierarchySoftmax, workers=c.workers,
				iter=c.iterNum, negative=c.negative
			)
		else:
			logger.info("Data size {} is larger than limit (max-memory-data-size: {}).	Dumping walks to disk.".format(data_size, c.maxMem))
			logger.info("Walking...")

			walks_filebase = self.WALKS_BASE_PATH
			walk_files = serialized_walks.write_walks_to_disk(
				G, walks_filebase, num_paths=c.num_walks,path_length=c.walkLength,
				alpha=0, rand=random.Random(c.seed),num_workers=c.workers
			)

			logger.info("Counting vertex frequency...")
			if not c.vertexFreqDegree:
				vertex_counts = serialized_walks.count_textfiles(walk_files, c.workers)
			else:
				# use degree distribution for frequency in tree
				vertex_counts = G.degree(nodes=G.iterkeys())

			logger.info("Training...")
			walks_corpus = serialized_walks.WalksCorpus(walk_files)
			model = Skipgram(
				sentences=walks_corpus, vocabulary_counts=vertex_counts, size=c.embed_size,
				window=c.window_size, min_count=0, trim_rule=None, workers=c.workers,
				sg=c.skipGram, hs=c.hierarchySoftmax, iter=c.iterNum, negative=c.negative
			)
		row_num = hpo_reader.get_hpo_num()
		self.hpo_embed = np.array([model.wv[str(i)] for i in range(row_num)])
		np.savez_compressed(self.EMBED_PATH, self.hpo_embed)
		c.save(self.CONFIG_JSON)




def get_embed(encoder_name):
	"""
	Returns:
		np.ndarray: shape=[hpo_num, vec_size]
	"""
	encoder = DeepwalkEncoder(encoder_name=encoder_name)
	return encoder.get_embed()


if __name__ == '__main__':
	pass
















