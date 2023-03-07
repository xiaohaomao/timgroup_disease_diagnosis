

import json
from core.utils.utils import get_file_list
from core.utils.constant import RESULT_PATH
d = {
	'TopAcc-1': 'TopAcc.1', 'TopAcc-5': 'TopAcc.5', 'TopAcc-10': 'TopAcc.10', 'TopAcc-20': 'TopAcc.20', 'TopAcc-50': 'TopAcc.50', 'TopAcc-100': 'TopAcc.100',
	'TopAcc@1': 'TopAcc.1', 'TopAcc@5': 'TopAcc.5', 'TopAcc@10': 'TopAcc.10', 'TopAcc@20': 'TopAcc.20', 'TopAcc@50': 'TopAcc.50', 'TopAcc@100': 'TopAcc.100'
}
json_paths = get_file_list(RESULT_PATH, lambda file_path: file_path.endswith('.json'))
for json_path in json_paths:
	result_dict = json.load(open(json_path))
	for oldName, new_name in d.items():
		if oldName in result_dict:
			result_dict[new_name] = result_dict[oldName]
			del result_dict[oldName]
	json.dump(result_dict, open(json_path, 'w'), indent=2)