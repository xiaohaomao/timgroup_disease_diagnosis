

import json
import os, shutil
from core.utils.constant import TEMP_PATH, PROJECT_PATH


def draw_base_func(temp_path, fig_path, para_dict):
	os.makedirs(temp_path, exist_ok=True)
	fig_dir = os.path.dirname(fig_path)
	os.makedirs(fig_dir, exist_ok=True)
	para_path = temp_path+os.sep+'para.json'
	json.dump(para_dict, open(para_path, 'w'))
	os.system('Rscript %(R_SCRIPT)s %(PARA_PATH)s' % {
		'R_SCRIPT': PROJECT_PATH+os.sep+'draw'+os.sep+'draw.R',
		'PARA_PATH': para_path
	})



def draw_multi_line_from_df(df, fig_path, x_col, y_col, class_col, class_order=None):
	"""
	Args:
		df (DataFrame): columns=(xName, yName, lineName)
		fig_path (str): 'price.png'
	"""
	temp_path = TEMP_PATH + os.sep + 'multiLineFromDF'
	os.makedirs(temp_path, exist_ok=True)
	csv_path = temp_path+os.sep+'df.csv'
	df.to_csv(csv_path)
	para_dict = {
		'drawFunc': 'draw_multi_line_from_df', 'csv_path': csv_path, 'fig_path': fig_path,
		'x_col': x_col, 'y_col': y_col, 'class_col': class_col, 'class_order': class_order
	}
	draw_base_func(temp_path, fig_path, para_dict)


def draw_multi_line(json_paths, fig_path, x_col, y_col, class_col, class_order=None):
	"""
	Args:
		json_paths (list of str): [filename1, ...]
		fig_path (str): path of figure file
	"""
	temp_path = TEMP_PATH+os.sep+'multiLine'
	para_dict = {
		'drawFunc': 'draw_multi_line', 'json_paths': json_paths, 'fig_path': fig_path,
		'x_col': x_col, 'y_col': y_col, 'class_col': class_col, 'class_order': class_order
	}
	draw_base_func(temp_path, fig_path, para_dict)


def draw_quartile_fig(json_paths, fig_path, x_col, y_col, class_col, class_order=None, x_order=None):
	"""
	Args:
		json_names (list of str): [filename1, ...]
		fig_path (str): path of figure file
	"""
	temp_path = TEMP_PATH+os.sep+'quartile'
	para_dict = {
		'drawFunc': 'draw_quartile', 'json_paths': json_paths, 'fig_path': fig_path,
		'x_col': x_col, 'y_col': y_col, 'class_col': class_col, 'class_order': class_order, 'x_order': x_order
	}
	draw_base_func(temp_path, fig_path, para_dict)



def draw_scatter(points, labels, fig_path, x_col='x', y_col='y', class_col='class', use_colour=False, show_label=True):
	"""
	Args:
		points (np.ndarray): shape=[point_num, 2]
		labels (np.ndarray): shape=[point_num]
		fig_path (str)
	"""
	data_dict = {x_col: points[:, 0], y_col: points[:, 1], class_col: labels}
	temp_path = TEMP_PATH+os.sep+'scatter'
	os.makedirs(temp_path, exist_ok=True)
	data_path = temp_path+os.sep+'data.json'
	json.dump(data_dict, open(data_path, 'w'))
	para_dict = {
		'drawFunc': 'draw_scatter', 'json_paths': [data_path], 'fig_path': fig_path,
		'x_col': x_col, 'y_col': y_col, 'class_col': class_col, 'use_colour': use_colour, 'show_label': show_label}
	draw_base_func(temp_path, fig_path, para_dict)


def draw_dodge_bar(json_paths, fig_path, x_col, y_col, class_col, class_order=None, x_order=None):
	temp_path = TEMP_PATH+os.sep+'dodgeBar'
	para_dict = {
		'drawFunc': 'draw_dodge_bar', 'json_paths': json_paths, 'fig_path': fig_path, 'x_col': x_col, 'y_col': y_col,
		'class_col': class_col, 'class_order': class_order, 'x_order': x_order,
	}
	draw_base_func(temp_path, fig_path, para_dict)



if __name__ == '__main__':
	import pandas as pd
	from core.utils.constant import TEMP_PATH
