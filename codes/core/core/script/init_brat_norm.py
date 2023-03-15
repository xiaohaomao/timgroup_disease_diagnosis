

from tqdm import tqdm

from core.reader.hpo_reader import HPOReader
from core.utils.utils import set_if_not_empty

hpo_reader = HPOReader()
hpo_dict = hpo_reader.get_hpo_dict()
chpo_dict = hpo_reader.get_chpo_dict()

key_order = ['ID', 'CNS_NAME', 'ENG_NAME', 'CNS_DEF', 'ENG_DEF', 'SYNONYM']
key_to_type_label = {
	'ID': '',
	'CNS_NAME': 'name:名称:',
	'ENG_NAME': 'name:Name:',
	'CNS_DEF': 'info:定义:',
	'ENG_DEF': 'info:Definition:',
	'SYNONYM': 'info:Synonyms:',
}

rows = []
for hpo in tqdm(hpo_dict):
	row_dict = {'ID': hpo, 'ENG_NAME': hpo_dict[hpo]['ENG_NAME']}
	set_if_not_empty(row_dict, 'ENG_DEF', chpo_dict.get(hpo, {}).get('ENG_DEF'))
	set_if_not_empty(row_dict, 'SYNONYM', chpo_dict.get(hpo, {}).get('SYNONYM'))
	set_if_not_empty(row_dict, 'CNS_NAME', chpo_dict.get(hpo, {}).get('CNS_NAME'))
	set_if_not_empty(row_dict, 'CNS_DEF', chpo_dict.get(hpo, {}).get('CNS_DEF'))
	rows.append('\t'.join(['{}{}'.format(key_to_type_label[k], row_dict[k]) for k in key_order if k in row_dict]) + '\n')

path = 'brat_hpo.txt'
with open(path, 'w') as f:
	f.writelines(rows)


