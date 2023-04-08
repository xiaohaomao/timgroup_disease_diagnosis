from core.node_embed.GCNEncoder import GCNDisAsLabelEncoder, GCNDisAsLabelConfig
from core.node_embed.GCNEncoder import GCNDisAsLabelFeatureEncoder, GCNDisAsLabelFeatureConfig
from core.node_embed.GCNEncoder import GCNDisAsFeatureEncoder, GCNDisAsFeatureConfig
from core.reader.hpo_reader import HPOReader
import os

import itertools

# ========================================================================
def process_dis_as_label(para):
	c, encoder_name = para
	encoder = GCNDisAsLabelEncoder(encoder_name=encoder_name)
	encoder.train(c)
	# print(encoder.get_embed(0)[0:3])

def train_gcn_dis_as_label_I():
	dis_num = HPOReader().get_dis_num()
	xtype_list = ['I']
	units_list = [64, 256]
	lr_list = [0.01]
	w_decay_list = [5e-6] #

	paras = []
	for xtype, units, lr, w_decay in itertools.product(xtype_list, units_list, lr_list, w_decay_list):
		c = GCNDisAsLabelConfig()
		c.xtype, c.units, c.lr, c.weight_decays = xtype, [units, dis_num], lr, [w_decay, 0.0]
		c.epoch_num = 100    # 200
		if xtype == 'W':
			c.xsize = units
		encoder_name = 'DisAsLabel_xt{}_units{}_lr{}_w_decay{}'.format(xtype, units, lr, w_decay)
		paras.append((c, encoder_name))

	for para in paras:
		process_dis_as_label(para)


def train_gcn_dis_as_label_W():
	dis_num = HPOReader().get_dis_num()
	xtype_list = ['W']
	units_list = [64, 256]
	lr_list = [0.01]
	w_decay_list = [5e-6]

	paras = []
	for xtype, units, lr, w_decay in itertools.product(xtype_list, units_list, lr_list, w_decay_list):
		c = GCNDisAsLabelConfig()
		c.xtype, c.units, c.lr, c.weight_decays = xtype, [units, dis_num], lr, [w_decay, 0.0]
		c.epoch_num = 100
		if xtype == 'W':
			c.xsize = units
		encoder_name = 'DisAsLabel_xt{}_units{}_lr{}_w_decay{}'.format(xtype, units, lr, w_decay)
		paras.append((c, encoder_name))

	for para in paras:
		process_dis_as_label(para)

# ========================================================================
def process_dis_as_label_feature(para):
	c, encoder_name = para
	encoder = GCNDisAsLabelFeatureEncoder(encoder_name=encoder_name)
	encoder.train(c)
	# print(encoder.get_embed(0)[0:3])

def train_gcn_dis_as_label_feature():
	dis_num = HPOReader().get_dis_num()
	units_list = [128, 256]
	lr_list = [0.01]
	w_decay_list = [5e-6]

	paras = []
	for units, lr, w_decay in itertools.product(units_list, lr_list, w_decay_list):
		c = GCNDisAsLabelFeatureConfig()
		c.units, c.lr, c.weight_decays = [units, units, dis_num], lr, [w_decay, w_decay, 0.0]
		c.layer_norm = [None, None, None]
		c.acts = ['relu', 'relu', None]
		c.keep_probs = [0.5, 0.5, 0.5]
		c.bias = [False, False, False]
		c.epoch_num = 100
		encoder_name = 'DisAsLabelFeature_layer3_units{}_lr{}_w_decay{}'.format(units, lr, w_decay)
		paras.append((c, encoder_name))

	for para in paras:
		process_dis_as_label_feature(para)


# ========================================================================

def process_dis_as_feature(para):
	c, encoder_name = para
	encoder = GCNDisAsFeatureEncoder(encoder_name=encoder_name)
	# encoder.train(c)
	print(encoder.get_embed(1, True)[0:3])

def train_gcn_dis_as_feature():
	dis_num = HPOReader().get_dis_num()
	units_list = [128]
	lr_list = [0.0001]
	w_decay_list = [0.0]

	paras = []
	for units, lr, w_decay in itertools.product(units_list, lr_list, w_decay_list):
		c = GCNDisAsFeatureConfig()
		c.units, c.lr, c.weight_decays = [units, dis_num], lr, [w_decay, 0.0]
		c.epoch_num = 200
		c.acts = ['sigmoid', None]
		encoder_name = 'DisAsFeature_sigmoid_units{}_lr{}_w_decay{}'.format(units, lr, w_decay)
		paras.append((c, encoder_name))

	for para in paras:
		process_dis_as_feature(para)




if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # "0" | "1"

	train_gcn_dis_as_feature()




