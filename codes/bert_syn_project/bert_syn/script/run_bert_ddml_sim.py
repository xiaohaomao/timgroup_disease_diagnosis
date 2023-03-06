

import os
import argparse
import pandas as pd
import json
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from bert_syn.core.bert_ddml_sim import BertDDMLConfig, BertDDMLSim
from bert_syn.core.data_helper import HPOReader
from bert_syn.utils.constant import RESULT_PATH, DATA_PATH
from bert_syn.utils.utils import write_standard_file

# e.g. python bert_syn/script/run_bert_ddml_sim.py --model_name albertTinyDDMLSim-AN_Hpo_N_SD20_BGD20_PC_C0-3-1024-32-fc1024 --gpu 0 --epoch 20 --lr 5e-5

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--model_name', type=str, default='BertDDMLSim')
	parser.add_argument('--epoch', type=int, default=-1)
	parser.add_argument('--from_last', type=int, default=0)
	parser.add_argument('--mode', type=str, default='train')    # 'train' | 'predict' | 'predict_hpo'
	parser.add_argument('--global_step', type=int, default=None)
	parser.add_argument('--multi_global_step', type=str, default=None)
	parser.add_argument('--save', type=int, default=1)
	parser.add_argument('--data', type=str, default='test') # 'eval' | 'test'
	parser.add_argument('--train_path', type=str, default=None)
	parser.add_argument('--tau', type=float, default=None)
	parser.add_argument('--lr', type=float, default=None)

	return parser.parse_args()


def train(args):
	config = BertDDMLConfig()
	if args.epoch > 0:
		config.num_train_epochs = args.epoch
	if args.train_path is not None:
		config.train_data_path = args.train_path
	if args.tau is not None:
		config.tau = args.tau
	if args.lr is not None:
		config.learning_rate = args.lr
	bert_sim = BertDDMLSim(args.model_name, config=config)
	bert_sim.set_dict_terms(HPOReader().get_cns_list())
	bert_sim.train(from_last=args.from_last)


def predict(args):
	bert_sim = BertDDMLSim(args.model_name)
	config = BertDDMLConfig()
	config.load(bert_sim.CONFIG_JSON)
	config.train_data_path = None
	config.eval_data_path = None
	data_type = args.data
	config.predict_data_path = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', data_type, 'ehr_terms_all.csv')
	config.predict_raw_to_true_json = os.path.join(DATA_PATH, 'preprocess', 'dataset', 'pumc', data_type, 'ehr_to_true_texts_all.json')
	bert_sim.restore(config, global_step=args.global_step)
	bert_sim.set_dict_terms(HPOReader().get_cns_list())
	bert_sim.predict(save_result_csv=args.save)


def predict_hpo(args):
	def gen_csv(src_term, pairs, scores, save_folder, global_step):
		os.makedirs(save_folder, exist_ok=True)
		src_term = src_term.replace('/', '_')
		pd.DataFrame(
			[{'text_a': text_a, 'text_b': text_b, 'label': score} for (text_a, text_b), score in zip(pairs, scores)],
			columns=['text_a', 'text_b', 'label']).sort_values('label', ascending=False).to_csv(
			os.path.join(save_folder, f'{src_term}-step{global_step}.csv'), index=False)

	if args.multi_global_step is not None:
		global_steps = args.multi_global_step.split(',')
		global_steps = [int(step) for step in global_steps]
	else:
		global_steps=[args.global_step]

	for global_step in global_steps:
		bert_sim = BertDDMLSim(args.model_name, user_mode=True)
		bert_sim.restore(global_step=global_step)
		tgt_terms = HPOReader().get_cns_list()
		bert_sim.set_dict_terms(tgt_terms)
		save_folder = os.path.join(bert_sim.RESULT_SAVE_FOLDER, 'user_terms')
		user_terms = [
			'甲床苍白',
			'尿频',
			'失钾性肾病',
			'头顶部胀痛',


			
			'呵呵',
			'有1姐姐',
			'走路有踩棉花感',
			'需依靠拐杖',
			'至天津总院就诊',
			'查头颅MRI未见异常'
			'否认家族性疾病及遗传病史',
			'患者无明显不适',
			'患者目前一般情况较好',
			'仍有轻微咳嗽',
		]
		for src_term in user_terms:
			score_ary, hpo_terms = bert_sim.predict_score(src_term)
			gen_csv(src_term, [(src_term, hpo_term) for hpo_term in hpo_terms], score_ary, save_folder, global_step)


def predict_best_hpo(args):
	# src_term_txt = os.path.join(DATA_PATH, 'preprocess', 'pumc_2000', 'naive_count-vocab.txt')
	src_term_txt = os.path.join(DATA_PATH, 'preprocess', 'pumc_2000', 'topwords-vocab.txt')
	src_terms = open(src_term_txt).read().strip().splitlines()

	if args.multi_global_step is not None:
		global_steps = args.multi_global_step.split(',')
		global_steps = [int(step) for step in global_steps]
	else:
		global_steps=[args.global_step]

	for global_step in global_steps:
		bert_sim = BertDDMLSim(args.model_name, user_mode=True)
		bert_sim.restore(global_step=global_step)
		tgt_terms = HPOReader().get_cns_list()
		bert_sim.set_dict_terms(tgt_terms)
		tgtterm_score_pairs = bert_sim.predict_best_match(src_terms)
		samples = [(src_term, tgt_term, score) for src_term, (tgt_term, score) in zip(src_terms, tgtterm_score_pairs) ]
		samples = sorted(samples, key=lambda item: item[2], reverse=True)
		save_txt = os.path.join(bert_sim.RESULT_SAVE_FOLDER, os.path.split(src_term_txt)[1])
		write_standard_file(samples, save_txt, split_char=' | ')


if __name__ == '__main__':
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	if args.mode == 'train':
		train(args)
	elif args.mode == 'predict':
		predict(args)
	elif args.mode == 'predict_hpo':
		predict_hpo(args)
	elif args.mode == 'predict_best_hpo':
		predict_best_hpo(args)
	else:
		raise RuntimeError('Unknown mode: {}'.format(args.mode))


