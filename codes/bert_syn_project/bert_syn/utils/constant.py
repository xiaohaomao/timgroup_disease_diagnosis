

import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC_PATH = os.path.join(PROJECT_PATH, 'bert_syn')
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')
TEMP_PATH = os.path.join(PROJECT_PATH, 'temp')
RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
BERT_INIT_MODEL_PATH = os.path.join(MODEL_PATH, 'bert', 'chinese_L-12_H-768_A-12')

JSON_FILE_FORMAT = 'JSON'
PKL_FILE_FORMAT = 'PKL'
NPY_FILE_FORMAT = 'NPY'
NPZ_FILE_FORMAT = 'NPZ'
SPARSE_NPZ_FILE_FORMAT = 'SPARSE_NPZ'
JOBLIB_FILE_FORMAT = 'JOBLIB'

RELU = 'relu'
LEAKY_RELU = 'leaky_relu'
TANH = 'tanh'
SIGMOID = 'sigmoid'
GELU = 'gelu'

TRAIN_DATA = 'train'
TEST_DATA = 'test'
VALIDATION_DATA = 'validation'

if __name__ == '__main__':
	print(PROJECT_PATH)
