import os

from core.helper.hyper.hyper_tune_helper import HyperTuneHelper
from core.utils.constant import get_tune_data_names, get_tune_data_weights, get_tune_metric_names, get_tune_metric_weights
from core.utils.constant import VALIDATION_DATA, TEST_DATA, RESULT_PATH


hyp_save_names = os.listdir(RESULT_PATH+'/hyper_tune')

for hyp_save_name in hyp_save_names:
	print(hyp_save_name)
	for eval_data in [TEST_DATA, VALIDATION_DATA]:
		hyper_helper = HyperTuneHelper(
			hyp_save_name,
			score_keys=[get_tune_data_names(eval_data), get_tune_metric_names()],
			score_weights=[get_tune_data_weights(eval_data), get_tune_metric_weights()],
			save_folder=RESULT_PATH + '/hyper_tune/{}/{}'.format(hyp_save_name, eval_data))
		hyper_helper.save_history()


