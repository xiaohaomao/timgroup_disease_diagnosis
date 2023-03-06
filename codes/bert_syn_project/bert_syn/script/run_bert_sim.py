

import os
import argparse
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from bert_syn.core.bert_sim import BertSimConfig, BertSim
from bert_syn.core.data_helper import HPOReader
from bert_syn.utils.constant import RESULT_PATH

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--model_name', type=str, default='BertSim')
	parser.add_argument('--epoch', type=int, default=3)
	parser.add_argument('--from_last', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')    # 'train' | 'predict' | 'predict_hpo'
	parser.add_argument('--global_step', type=int, default=None)
	parser.add_argument('--multi_global_step', type=str, default=None)

	return parser.parse_args()


def train(args):
	config = BertSimConfig()
	config.num_train_epochs = args.epoch
	bert_sim = BertSim(args.model_name, config=config)
	bert_sim.train(from_last=args.from_last)


def predict(args):
	bert_sim = BertSim(args.model_name)
	config = BertSimConfig()
	config.load(bert_sim.CONFIG_JSON)
	config.train_data_path = None
	config.eval_data_path = None
	bert_sim.restore(config, global_step=args.global_step)
	bert_sim.predict()


def predict_hpo(args):
	def make_pairs(src_term, tgt_terms):
		return [(src_term, tgt_term) for tgt_term in tgt_terms]
	def gen_csv(src_term, pairs, probs, save_folder, global_step):
		os.makedirs(save_folder, exist_ok=True)
		src_term = src_term.replace('/', '_')
		pd.DataFrame(
			[{'text_a': text_a, 'text_b': text_b, 'label': prob} for (text_a, text_b), prob in zip(pairs, probs)],
			columns=['text_a', 'text_b', 'label']).sort_values('label', ascending=False).to_csv(
			os.path.join(save_folder, f'{src_term}-step{global_step}.csv'), index=False)

	if args.multi_global_step is not None:
		global_steps = args.multi_global_step.split(',')
		global_steps = [int(step) for step in global_steps]
	else:
		global_steps=[args.global_step]

	for global_step in global_steps:
		bert_sim = BertSim(args.model_name, user_mode=True)
		bert_sim.restore(global_step=global_step)
		tgt_terms = HPOReader().get_cns_list()
		save_folder = os.path.join(bert_sim.RESULT_SAVE_FOLDER, 'user_terms')
		user_terms = [
			# In training set
			'鼻咽肿瘤', '特发性假性肠梗阻', '心电图异常', '肥胖症', '无牙', '情绪不稳定', #
			# In eval set
			'增大子宫', '点彩状骨骺', '耳部肿瘤', '汗过少，少汗', #
			# User select
			'出牙过早', '呼吸衰竭', '扁桃体炎', '尺桡关节脱位', '甲状腺功能亢进', '血压升高', '意识模糊',
		]
		for src_term in user_terms:
			pairs = make_pairs(src_term, tgt_terms)
			probs = bert_sim.predict_probs(pairs)
			gen_csv(src_term, pairs, probs, save_folder, global_step)


if __name__ == '__main__':
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	if args.mode == 'train':
		train(args)
	elif args.mode == 'predict':
		predict(args)
	elif args.mode == 'predict_hpo':
		predict_hpo(args)
	else:
		raise RuntimeError('Unknown mode: {}'.format(args.mode))
