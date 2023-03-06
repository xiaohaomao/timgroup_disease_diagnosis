

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  #
plt.rcParams['axes.unicode_minus'] = False  #

import seaborn as sns
sns.set(font='SimHei')  #
sns.set_style('whitegrid', {'font.sans-serif':['simhei','Arial']})

import pandas as pd


def simple_line_plot(figpath, x, y, x_label=None, y_label=None, title=None):
	"""
	Args:
		x (array-like)
		y (array-like)
	"""
	x_label = 'x' if x_label is None else x_label
	y_label = 'y' if y_label is None else y_label
	title = 'Simple Line Plot' if title is None else title
	df = pd.DataFrame({x_label: x, y_label: y})
	ax = plt.axes()
	sns.lineplot(x=x_label, y=y_label, data=df, ax=ax)
	ax.set_title(title)
	plt.savefig(figpath)
	plt.close()


def simple_multi_line_plot(figpath, x_list, y_list, line_names=None, x_label=None, y_label=None, title=None, figsize=None):
	"""
	Args:
		x_list (list): [array-like, ...]
		y_list (list): [array-like, ...]
	"""
	line_num = len(x_list)
	line_names = ['line'+str(i) for i in range(line_num)] if line_names is None else line_names
	x_label = 'x' if x_label is None else x_label
	y_label = 'y' if y_label is None else y_label
	title = 'Simple Line Plot' if title is None else title
	df = pd.concat([pd.DataFrame({x_label: x_list[i], y_label: y_list[i], 'lines': line_names[i]}) for i in range(line_num)])
	plt.figure(figsize=figsize)
	ax = plt.axes()
	sns.lineplot(x=x_label, y=y_label, hue='lines', data=df, ax=ax)
	ax.set_title(title)
	plt.savefig(figpath)
	plt.close()


def simple_dot_plot(figpath, x, y, p_types=None, p_type_order=None, sizes=None, markers=None, pid2text=None,
		x_label=None, y_label=None, title=None, figsize=None, p_type_label=None, p_size_label=None, p_style_label=None, palette=None):
	"""
	Args:
		x (array-like)
		y (array-like)
		p_types (array-like): different point color for different pType
		p_type_order (larray-like):
		sizes (dict or list or tuple or None): {label: size}
		markers (dict or list or None): {label: marker}
		pid2text (dict): {pid: str}
	"""
	x_label = 'x' if x_label is None else x_label
	y_label = 'y' if y_label is None else y_label
	title = 'Simple Dot Plot' if title is None else title
	pid2text = {} if pid2text is None else pid2text
	p_type_label = 'type' if p_type_label is None else p_type_label

	df_dict = {x_label: x, y_label: y}
	if p_types is not None:
		df_dict[p_type_label] = p_types
	df = pd.DataFrame(df_dict)

	plt.figure(figsize=figsize)
	ax = plt.axes()
	fig = sns.scatterplot(x=x_label, y=y_label, hue=p_type_label, size=p_size_label, style=p_style_label,
		hue_order=p_type_order, sizes=sizes, markers=markers, data=df, ax=ax, palette=palette)
	for pid, p_text in pid2text.items():
		fig.text(x[pid]+0.02, y[pid], p_text, horizontalalignment='left', size='medium', color='black', weight='semibold')
	ax.set_title(title)

	plt.savefig(figpath)
	plt.close()


def simple_dist_plot(figpath, x, bins, x_label=None, title=None, figsize=None, x_lim=(None, None)):
	x_label = 'x' if x_label is None else x_label
	title = 'Simple Dist Plot' if title is None else title
	plt.figure(figsize=figsize)
	ax = plt.axes()
	sns.distplot(x, bins=bins, kde=False, rug=False, axlabel=x_label, ax=ax)
	ax.set_title(title)
	ax.set_xlim(left=x_lim[0], right=x_lim[1])
	plt.savefig(figpath)
	plt.close()


def simple_heat_map(figpath, a, row_labels, col_labels, row_name=None, col_name=None, title=None):
	"""
	Args:
		a (np.ndarray): 2d array
		row_labels (list)
		col_labels (list)
	"""
	row_name = 'y' if row_name is None else row_name
	col_name = 'y' if col_name is None else col_name
	title = 'Heat Map' if title is None else title
	df = pd.DataFrame(a, index=row_labels, columns=col_labels)
	ax = plt.axes()
	sns.heatmap(df, ax=ax)
	ax.set_xlabel(col_name); ax.set_ylabel(row_name); ax.set_title(title)
	plt.savefig(figpath)
	plt.close()


if __name__ == '__main__':
	from core.utils.constant import TEMP_PATH
	import numpy as np

	simple_heat_map(TEMP_PATH+'/heatmap.jpg', np.random.randn(10, 12), list(range(2, 2+10)), list(range(3, 3+12)), 'row', 'col', 'hh')