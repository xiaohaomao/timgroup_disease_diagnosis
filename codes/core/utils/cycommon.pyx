# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True

"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()

ctypedef unsigned int uint
ctypedef np.int32_t INT
ctypedef np.int64_t LONG_INT
ctypedef np.float64_t DOUBLE
ctypedef np.float32_t FLOAT

def to_rank_score(np.ndarray[DOUBLE, ndim=2] score_mat, np.ndarray[INT, ndim=2] arg_mat):
	"""
	Args:
		score_mat (np.ndarray): (modelNum, disNum)
	Returns:
		np.ndarray
	"""
	cdef:
		int i
		int j
		int col
		DOUBLE EPS = 1e-12
		DOUBLE diff
		DOUBLE score = 0.0
		DOUBLE score_step = 1.0
		DOUBLE last_raw_score
		int row_num = score_mat.shape[0]
		int col_num = score_mat.shape[1]

	for i in range(row_num):
		last_raw_score = score_mat[i, arg_mat[i, 0]]
		for j in range(1, col_num):
			col = arg_mat[i, j]
			diff = score_mat[i, col] - last_raw_score
			last_raw_score = score_mat[i, col]
			if not (diff < EPS and diff > -EPS) :
				score += score_step
			score_mat[i, col] = score
		score_mat[i, arg_mat[i, 0]] = 0.0
		score_step /= col_num
		score = 0.0




