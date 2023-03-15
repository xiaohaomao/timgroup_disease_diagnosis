
import re
import os
from tqdm import tqdm
import json
from copy import deepcopy

from core.text_handler.max_inv_text_searcher import MaxInvTextSearcher
from core.text_handler.text_searcher import TextSearcher
from core.text_handler.term_matcher import ExactTermMatcher, BagTermMatcher
from core.reader import HPOReader, HPOFilterDatasetReader
from core.utils.constant import RESULT_PATH, DATA_PATH
from core.patient.cjfh_patient_generator import CJFHPatientGenerator
from core.patient.patient_generator import PatientGenerator
from core.text_handler.syn_generator import UMLSSynGenerator
from core.explainer.search_explainer import SearchExplainer
from core.utils.utils import str_list_product, dict_list_extend, dict_list_combine, split_path
from core.explainer.dataset_explainer import LabeledDatasetExplainer

invalid_end = {'+'}
search_skip = {'眼震:'}

neg_rules = [
	'脊髓',   # -> 脊髓病
	'眼水',   # -> 水眼
	'前开',   # -> 前牙开咬
	'扭转',   # -> 肠扭转
	'痉挛',   # -> 癫痫发作
	'发作',   # -> 癫痫发作
	'作发',   # -> 癫痫发作
	'关节',   # -> 关节病、关节炎
	'模糊',   # -> 晕厥
	'一个',   # -> 无虹膜
	'甲状腺', # -> 甲状腺异常
	'腺甲状',  # -> 甲状腺异常
	'唇',    # -> 上唇裂
	'僵硬',   # -> 肌肉僵硬
	'震:眼',  # -> 眼震
	'发白',    # -> 白发
	'热发',   # -> 发热
	'夜尿',   # -> 夜尿症
	'期睡眠', # -> 困倦
	'头晕', # -> 眩晕
	'尿无', # -> 无尿症
	'型糖尿病', # -> 1型/2型糖尿病
	'食多', # -> 少食多餐
	'小脑脑干无明显萎缩',
	'肿瘤标志物', # -> 肿瘤
]

neg_prules = [
	'反射.{0,3}活跃', # -> 反射活跃
]

pos_rules = [
	# 运动; 协调
	(['无法自行行走', ], 'HP:0002540'), # 行走不能
	(['小碎步', '步距小'], 'HP:0007311'),  # 小碎步步态
	(['步基宽', '步基略宽', '步基稍宽'], 'HP:0002136'),  # 宽基步态
	(['前屈姿势'], 'HP:0002533'),   # 姿势异常

	# 眼
	(['视物成双', '双影'], 'HP:0000651'), # 复视
	(['眼睑后退', '脸睑退缩'], 'HP:0000492'), # 眼睑异常
	(['眼睑退缩'], 'HP:0030802'),    # 下眼睑退缩
	(['眼急动'], 'HP:0000570'),    # 眼球运动(saccadic eye movements)异常

	# 肌肉
	(['面具脸'], 'HP:0000298'),    # 面具般面容
	(['表情少', '表情少'], 'HP:0004673'),    # 面部表情减少
	(['表情淡漠', '表情呆板'], 'HP:0005329'), # 面部表情木讷
	(['癫痫'], 'HP:0001250'),  # 癫痫发作
	(['发作性抽搐'], 'HP:0001250'),    # 癫痫发作
	(['抽搐'], 'HP:0100033'), # 抽动症
	(['四肢远端肌肉萎缩'], 'HP:0003693'),  # 远端肌肉萎缩
	(['面部肌颤搐', '面部抽搐'], 'HP:0000317'),    # 面肌纤维颤搐
	(['抽筋'], 'HP:0003394'),   # 肌肉痉挛
	(['舌肌萎缩'], 'HP:0012473'),   # 舌萎缩
	(['肌束颤'], 'HP:0002380'),    # 肌束震颤
	(['肌肉萎缩'], 'HP:0003202'), # 肌萎缩

	# 脑
	(['枕大池囊肿'], 'HP:0100702'),  # 蛛网膜囊肿
	(['语言轻度障碍', '小脑语言', '共济失调语言'], 'HP:0002463'),   # 语言障碍
	(['延髓萎缩'], 'HP:0011441'), # 延髓异常

	# 泌尿
	(['遗尿'], 'HP:0000805'), # 遗尿症

	# 其他
	(['纳差'], 'HP:0004396'),   # 食欲不振
	(['腰间盘突出', '腰间盘脱出'], 'HP:0100712'),  # 腰椎脊柱异常

	# ([], ''),
	# ([], ''),
	# ([], ''),
	# ([], ''),
]

pos_prules = [
	# 病理征
	(['病理征.{0,3}\+'], 'HP:0007256'), # 锥体功能障碍
	(['(Babinski‘s sign|巴氏征|Babinski|Babinski征).{0,3}\+'], 'HP:0003487'), # 巴彬斯基征
	(['Chaddock征.{0,3}\+', '(可疑|引出).{0,3}Chaddock征'], 'HP:0010875'),   # 查多克反射

	# 运动; 协调
	(['(快速)?轮替(动作)?.{0,3}(左侧|右侧)?(轻度|中度)?(笨|笨拙|欠佳|差|不稳|异常)'], 'HP:0002075'),  # 轮替运动障碍
	(['步态:.*?(差|不稳|拖步|搀扶)', '(不能|无法|难以)?走直线.{0,3}(不能|不好|困难|步态差|异常)?', '(前冲|前屈)(步态|体态|姿势)', ' 联带动作(消失|少|减少)', '无联带动作', '转身动作分解'], 'HP:0002317'), # 步态不稳
	(['辨距不良'], 'HP:0001310'), # # 辨距不良
	(['拖步'], 'HP:0002362'), # 曳行步态
	(['缺乏平衡感', '平衡感?(差|不好)', '(易失去|不能掌握)平衡'], 'HP:0002141'), # 步态失衡
	(['并足站立.{0,3}闭目.{0,3}(不稳|尤甚|摇晃|无法)', '(Romberg征|闭目难立征).{0,3}(闭眼|闭目)?(并足站|站立)?.{0,3}(困难|不能|不稳|不好|\+).{0,3}(闭眼|闭目)?', 'Romberg征.{0,3}(无法|不能).{0,3}(并足站立)'], 'HP:0002403'),   # 闭目难立征阳性
	(['站立.{0,3}(不稳|摇晃|后退)', '(不能|无法).{0,3}(双足并|并足)(站立|立站)(行走)?', '并足站不能', '(扶持|搀扶)站立'], 'HP:0003698'),   # 站立困难
	(['(无法|不能)(自行|独立)?(站立)?行走'], 'HP:0002540'), # 行走不能
	(['搀扶行走', '迈步困难', '(走路|行走).{0,3}(搀扶|扶持|困难)'], 'HP:0002355'), # 行走困难
	(['(运动|行动|动作|转身).{0,3}(迟缓|缓慢)'], 'HP:0002067'), # 运动迟缓
	(['(饮水|吞咽|进食|吃饭).{0,3}(呛|呛咳|发呛|费力|费劲|噎|困难)', '呛咳'], 'HP:0002015'),   # 吞咽困难
	(['(容易|易).{0,2}(摔|跌)(跤|倒|伤)?'], 'HP:0002359'), # 频繁跌倒
	(['(容易|易).{0,2}(摔|跌)(跤|倒|伤)?', '走路不稳', '醉酒感'], 'HP:0012651'), # 姿势不稳
	(['走路不稳', '醉酒感'], 'HP:0012651'), # 行走不稳
	([]+str_list_product(['跟膝胫', '指鼻'], [':?((左|右)侧)?.{0,3}'], ['(明显|严重|中度|轻度|略|稍)?(不|欠)(稳|准|稳准|佳)', '(无法做|双侧差|笨|笨拙)']), 'HP:0100660'),   # 运动障碍
	(['(强|控制不住)苦?笑'], 'HP:0000748'), # 不适宜的发笑
	(['强哭'], 'HP:0030215'), # 不当哭泣
	(['麻木'], 'HP:0003474'), # 感觉损害
	(['(动作|运动)(慢|迟缓)'], 'HP:0002067'),  # 运动迟缓
	(['精细动作.{0,3}(差|异常)', '不能做精细动作', '(写字|字迹|持筷).{0,3}(困难|差|不好|改变|变化大|不灵活|欠灵活|不清|发抖|不稳)'], 'HP:0007010'),  # 精细动作协调差

	# 肌肉
	(['(肌)?张力(四肢|右肢|左肢|下肢|上肢|肢体)?.{0,4}(下降|偏低|低下|低)'], 'HP:0001252'),    # 肌张力减退
	(['(肌)?张力(四肢|右肢|左肢|下肢|上肢)?.{0,4}(增|偏|稍)?(?<!不)高'], 'HP:0001276'),   # 肌张力增高
	(['下肢.{0,4}肌张力.{0,3}(增|偏|稍)?(?<!不)高', '肌张力.{0,4}下肢.{0,3}(增|偏|稍)?(?<!不)高'], 'HP:0006895'),   # 下肢肌张力增高
	(['上肢.{0,4}肌张力.{0,3}(增|偏|稍)?(?<!不)高', '肌张力.{0,4}上肢.{0,3}(增|偏|稍)?(?<!不)高'], 'HP:0200049'),   # 上肢肌张力增高
	(['(上肢|手)肌肉?萎缩'], 'HP:0009129'),   # 上肢肌萎缩
	(['(下肢|脚|腿)肌肉?萎缩'], 'HP:0007210'),   # 下肢肌萎缩
	(['(面)(部|肌)?抽搐'], 'HP:0000317'),   # 面肌纤维颤搐
	(['腿抽搐'], 'HP:0001281'), # 手足抽搐
	(['(腿|下肢|行走).{0,3}(发沉|无力|软|乏力)'], 'HP:0007340'), # 下肢肌肉无力
	(['(上肢|手).{0,3}(发沉|无力|乏力)', '持物.{0,3}(不稳|无力|乏力)'], 'HP:0003484'), # 上肢肌无力
	(['(肢体|全身|四肢|浑身).{0,3}(发沉|无力|乏力)'], 'HP:0003690'), # 四肢肌肉无力
	(['(肢体|肢|上肢|下肢|四肢|肌|肌力|肌肉|右侧|左侧).{0,3}(不灵活|发僵|僵硬)'], ['HP:0003552', 'HP:0002063']),   # 肌肉僵硬; 强直
	(['(手|手部).{0,3}(抖|抖动|震颤)'], 'HP:0002378'), # 手部震颤
	(['(头|头部).{0,3}(抖|抖动|震颤)'], 'HP:0002346'), # 头部震颤
	(['肢体抽动'], 'HP:0100033'), # 抽动症

	# 反射
	(['双?(四|上)肢腱?反射(未引出|消失|阴性)'], 'HP:0012046'), # 上肢反射消失
	(['双?(四|下)肢腱?反射(未引出|消失|阴性)'], 'HP:0002522'), # 下肢反射消失
	(['膝腱?反射.{0,3}(未引出|消失|阴性)'], 'HP:0006844'),   # 膝反射消失
	(['腱反射.{0,3}(弱|低|低)'], 'HP:0001265'),   # 腱反射减弱
	(['膝腱?反射.{0,3}(弱|低)'], 'HP:0011808'), # 膝反射减弱
	(['双?(四|下)肢腱反射?.{0,3}(弱|低)'], 'HP:0002600'),   # 下肢反射减弱
	(['腱反射.{0,3}(亢进|\+\+)'], 'HP:0006801'),   # 腱反射亢进
	(['膝腱?反射(亢进|\+\+)'], 'HP:0007083'), # 膝腱反射亢进
	(['双?(四|下)肢腱反射(亢进|\+\+)'], 'HP:0002395'), # 下肢腱反射亢进
	(['(踝|跟腱)反射.{0,3}未引出'], 'HP:0200101'), # 踝反射降低/缺如
	(['(吸吮反射|吸吮).{0,3}\+?'], 'HP:0030906'), # 吸吮反射
	(['(掌心下颌).{0,3}\+?'], 'HP:0030902'), # 掌颌反射
	(['胃食管(返|反)流'], 'HP:0002020'),  # 胃食管反流

	# 眼
	(['(眼动|眼球位置|双眼|眼球|眼).{0,3}外展.{0,3}(受限|欠充分|不能|不充分|困难|不全|露白|不到边)', '(眼|眼动|眼球活动|上视|侧视|下视|上下视).{0,3}(受限|欠充分|不能|不充分|困难|不全)'] , 'HP:0007941'),    # 有限的眼运动
	(['慢眼动', '辐辏运动差', '(眼动|眼球位置):(眼球活动)?.{0,3}(慢)'] , 'HP:0000496'),    # 眼球运动(eye movement)异常
	(['平滑跟踪.{0,4}(慢|差|分裂)'], 'HP:0007668'), # 视物追踪障碍
	(['(细小|粗大)?眼震.{0,4}垂直', '垂直眼震', '垂直.{0,4}(细小|粗大)?眼震'], 'HP:0010544'), # 垂直眼球震颤
	(['(细小|粗大)?眼震.{0,4}水平', '水平.{0,4}(细小|粗大)?眼震'], 'HP:0000666'), # 水平眼震
	(['突眼', '眼球位置:突出', '眼球前突'], 'HP:0000520'),   # 眼球突出
	(['眼睑.{0,3}(束颤|抽搐|颤搐|跳动)'], 'HP:0030826'),   # 眼睑震颤
	(['视物(不清|模糊)'], 'HP:0000505'), # 视觉障碍
	(['(视力|视觉).{0,3}(下降|减退)'], 'HP:0000572'), # 视力下降

	# 脑
	(['(脑干)?小脑(脑干)?.{0,3}萎缩'], 'HP:0001272'), # 小脑萎缩
	(['(小脑)?脑干(小脑)?.{0,3}萎缩'], 'HP:0007366'), # 影响脑干的萎缩/退化
	(['脑干.{0,3}小脑.{0,3}萎缩', '小脑.{0,3}脑干.{0,3}萎缩'], ['HP:0001272', 'HP:0007366']), # 小脑萎缩; 影响脑干的萎缩/退化
	(['(记忆力|记忆).{0,3}(下降|减退)', '健忘'], 'HP:0002354'), # 记忆障碍
	(['皮层.{0,3}(萎缩|不饱满|欠饱满)'], 'HP:0002120'), # 大脑皮层萎缩
	(['脑桥.{0,3}萎缩'], 'HP:0006879'), # 脑桥小脑萎缩
	(['基底节.{0,3}白质异常'], ['HP:0002134', 'HP:0012751', 'HP:0002500']), # 基底节异常; 核磁共振基底节异常信; 脑白质异常
	(['脑?白质(异常|病变)'], 'HP:0002500'), # 脑白质异常
	(['白质点状长T2信号'], 'HP:0030081'), # 点状脑室周围白质 T2 高信号灶
	(['脑白质脱髓鞘'], 'HP:0002352'), # 白质脑病
	(['顶叶.{0,3}(萎缩|不饱满|欠饱满)'], 'HP:0012104'), # 顶叶皮质萎缩
	(['额顶?叶.{0,3}(萎缩|不饱满|欠饱满)'], 'HP:0006913'), # 额叶皮质萎缩
	(['脑室(扩大|扩张)'], 'HP:0002119'), # 巨脑室
	(['智力(发育差|减退|下降)'], ['HP:0001249']), # 智力残疾
	(['(大脑)?皮层下?.{0,2}(点状|小片状)?异常(信号)?'], 'HP:0002538'), # 大脑皮层异常
	(['计算力.{0,3}(下降|差)'], 'HP:0001268'), # 智能衰退

	# 语言; 声音
	([]+str_list_product(['(言语|语言|吐字|构音|说话|口齿|咬字).{0,3}'], ['(不|欠)(清|清楚|流利|利)', '(含混|含糊)']), 'HP:0001350'),   # 言语不清
	(['爆破语言', '爆破性语言'], 'HP:0002168'), # 断续言语（Scanning speech；Explosive speech）
	(['(声音|嗓音)(嘶哑|沙哑)'], 'HP:0001609'), # 声音嘶哑
	(['(声|音|声音|嗓音|音量|说|说话|话)(低|低微|小)'], 'HP:0001621'), # 嗓音微弱
	(['(声音|嗓音)(改变|低)'], 'HP:0001608'), # 声音异常

	# 睡眠
	(['打鼾.{0,3}(有时|频繁|明显)?(睡|睡眠)中?呼吸暂停'], 'HP:0002870'), # 阻塞性睡眠呼吸暂停
	(['打鼾'], 'HP:0002360'), # 睡眠障碍
	(['(入睡|睡|眠|睡眠).{0,3}(差|难|困难|不好|欠佳)'], 'HP:0100785'), # 失眠
	(['(梦|睡|夜).{0,3}(喊|叫|肢体挥动|说话|呻吟|动作多)'], 'HP:0002360'),  # 睡眠障碍
	(['(梦|睡|夜).{0,3}(喊|叫|肢体挥动)'], 'HP:0030765'), # 睡惊症
	(['(夜|睡|睡眠|梦)(中|里)?.{0,3}(脸|肢|下肢|腿).{0,3}(抽动|挥动)'], 'HP:0012323'), # 睡眠肌阵挛

	# 泌尿系统; 生殖系统; 排便
	(['(小便|尿|排尿).{0,3}(费劲|费力|无力|障碍|困难)', '排尿时间延长', '尿不(尽|净)', '残余尿', '尿残余'], ['HP:0000019', 'HP:0100518']),    # 排尿困难（尿不净不等于尿潴留）
	(['(小便|尿)(稍|较前)?(次数多|频)', '尿频急', '尿频尿失禁'], 'HP:0100515'),    # 尿频
	(['(二便|小便|尿).{0,3}(控制不好|失禁)', '尿裤子'], 'HP:0000020'), # 尿失禁
	(['夜尿(次数)?(多|增多|很多)', '夜尿(?!(1|2)).{0,3}次'], 'HP:0000017'),   # 夜尿症
	(['(小便|尿)频?急'], 'HP:0000012'),    # 尿急
	(['(排便|二便|大便).{0,3}(控制不好|失禁)'], 'HP:0002607'),   # 大便失禁
	(['便秘', '(排|大)便.{0,3}(无力|障碍)'], 'HP:0002019'), # 便秘
	(['(大|二)?便急'], 'HP:0012701'), # 排便紧迫感
	(['性功能.{0,3}(减退|障碍|下降)'], 'HP:0000144'),  # 生育能力下降
	(['残余尿'], 'HP:0002839'), # 膀胱括约肌功能障碍

	# 排汗
	(['少出?汗', '出?汗少', '(无汗|不出汗)'], 'HP:0000966'), # 少汗症
	(['(无汗|不出汗)'], 'HP:0000970'), # 无汗症
	(['多出?汗', '出?汗多', '出?虚汗'], 'HP:0000975'), # 多汗症

	# 血
	(['血压.{0,3}(偏)?低', '低血压', ], 'HP:0002615'), # 低血压
	(['血钾偏?低'], 'HP:0002900'), # 低钾血症
	(['血脂.{0,3}高', '高血脂'], 'HP:0003077'), # 高脂血症

	# 其他
	(['消瘦', '体重.{0,3}(下降|减轻)'], 'HP:0004325'), # 体重下降
]


def get_hpo_reader():
	return HPOFilterDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'])


def neg_rules_to_dict(nr_list):
	return {None: nr_list}


def pos_rules_to_dict(pr_list):
	ret_dict = {}
	for term_list, code_item in pr_list:
		if isinstance(code_item, str):
			dict_list_extend(code_item, term_list, ret_dict)
		else:
			assert isinstance(code_item, list)
			for code in code_item:
				dict_list_extend(code, term_list, ret_dict)
	return ret_dict


def save_hpo_patients(search_results, diag_list, json_path):
	assert len(search_results) == len(diag_list)
	pg = PatientGenerator(hpo_reader=get_hpo_reader())
	patients = [[pg.process_pa_hpo_list(sr[0], reduce=False), dis_codes] for sr, dis_codes in zip(search_results, diag_list) ]
	explainer = LabeledDatasetExplainer(patients)
	folder, name, postfix = split_path(json_path)
	json.dump(explainer.explain(), open(os.path.join(folder, name+'_stat.json'), 'w'), indent=2, ensure_ascii=False)
	json.dump(patients, open(json_path, 'w'), indent=2, ensure_ascii=False)


def show_snomed_gu():
	pg = CJFHPatientGenerator(hpo_reader=get_hpo_reader())
	text_list, search_results = pg.get_man_search_text_result()
	output_txt = RESULT_PATH + '/text_handle/CJFH/1-lxh.txt'
	SearchExplainer(text_list, search_results).explain_save_txt(output_txt)


def gen_hpo_patient(searcher, output_txt, patient_json, out_brat_folder):
	pg = CJFHPatientGenerator(hpo_reader=get_hpo_reader())
	text_list, diag_list = zip(*pg.get_text_patients())
	text_pids = pg.get_text_pids()
	pid2diag = pg.get_pid_to_diag_str()
	print('searching:', searcher.name)
	search_results = [searcher.search(text) for text in tqdm(text_list)]

	explainer = SearchExplainer(text_list, search_results)
	os.makedirs(out_brat_folder, exist_ok=True)
	for pid, text in zip(text_pids, text_list):
		text = text + '\n===========================\n' + pid2diag[pid]
		open(os.path.join(out_brat_folder, '{:04}.txt'.format(pid)), 'w').write(text)
	out_brat_anns = [os.path.join(out_brat_folder, '{:04}.ann'.format(pid)) for pid in text_pids]
	explainer.write_as_anns(out_brat_anns)
	explainer.explain_save_txt(output_txt)
	save_hpo_patients(search_results, diag_list, patient_json)


def search_with_dict(syn_dict, syn_dict_name, gold_to_hpo=None, gold_p_to_hpo=None):
	matchers = [
		BagTermMatcher(syn_dict, syn_dict_name, gold_to_hpo, gold_p_to_hpo),
		ExactTermMatcher(syn_dict, syn_dict_name, gold_to_hpo, gold_p_to_hpo)
	]
	for matcher in matchers:
		searcher = MaxInvTextSearcher(matcher, invalid_end, search_skip)
		gen_hpo_patient(
			searcher=searcher,
			output_txt=os.path.join(DATA_PATH, 'raw', 'CJFH', 'cjfh_auto_tag', 'CJFH-{}.txt'.format(searcher.name)),
			patient_json=os.path.join(DATA_PATH, 'preprocess', 'patient', 'CCRD_OMIM_ORPHA', 'CJFH', '{}-patients.json'.format(searcher.name)),
			out_brat_folder = os.path.join(DATA_PATH, 'raw', 'CJFH', 'auto_tag', 'CJFH-{}'.format(searcher.name)),
		)


def only_hpo():
	search_with_dict({}, 'HPODict')


def hpo_source_umls():
	search_with_dict(UMLSSynGenerator().get_hpo_to_source_syn_terms(), 'SourceSynDict')


def hpo_bg_umls():
	search_with_dict(UMLSSynGenerator().get_hpo_to_syn_terms_with_bg_evaluate(), 'BGEvaSynDict', neg_rules_to_dict(neg_rules))


def hpo_all_umls():
	search_with_dict(UMLSSynGenerator().get_hpo_to_syn_terms(), 'SimSynDict')


def hpo_umls_rules():
	sg = UMLSSynGenerator()
	run_syn_dict_rules(sg.get_hpo_to_syn_terms_with_bg_evaluate(), 'BGEvaSynDict')


def hpo_source_umls_rules():
	sg = UMLSSynGenerator()
	run_syn_dict_rules(sg.get_hpo_to_source_syn_terms(), 'SourceSynDict')


def run_syn_dict_rules(syn_dict, syn_dict_name):
	hpo_to_golds = dict_list_combine(neg_rules_to_dict(neg_rules), pos_rules_to_dict(pos_rules))
	hpo_to_gold_ps = dict_list_combine(neg_rules_to_dict(neg_prules), pos_rules_to_dict(pos_prules))

	matcher = BagTermMatcher(syn_dict, syn_dict_name, hpo_to_golds, hpo_to_gold_ps)
	searcher = MaxInvTextSearcher(matcher, invalid_end, search_skip)
	gen_hpo_patient(
		searcher=searcher,
		output_txt=os.path.join(DATA_PATH, 'raw', 'CJFH', 'auto_tag', 'CJFH-{}-rules.txt'.format(searcher.name)),
		patient_json=os.path.join(DATA_PATH, 'preprocess', 'patient', 'CCRD_OMIM_ORPHA', 'CJFH', '{}-rules-patients.json'.format(searcher.name)),
		out_brat_folder=os.path.join(DATA_PATH, 'raw', 'CJFH', 'auto_tag', 'CJFH-{}-rules'.format(searcher.name)),
	)


def test_all(text_list, folder_path, gold_to_hpo=None, gold_p_to_hpo=None):
	os.makedirs(folder_path, exist_ok=True)
	sg = UMLSSynGenerator()
	all_syn_dict = {
		'SimSynDict': sg.get_hpo_to_syn_terms(),
		'BGEvaSynDict': sg.get_hpo_to_syn_terms_with_bg_evaluate(),
		'SourceSynDict': sg.get_hpo_to_source_syn_terms(),
	}

	matchers = []
	for syn_dict_name, syn_dict in all_syn_dict.items():
		matchers.append(ExactTermMatcher(syn_dict, syn_dict_name, gold_to_hpo, gold_p_to_hpo))
		matchers.append(BagTermMatcher(syn_dict, syn_dict_name, gold_to_hpo, gold_p_to_hpo))
	searchers = []
	for matcher in matchers:
		searchers.append(MaxInvTextSearcher(matcher, invalid_end, search_skip))

	for searcher in searchers:
		output_txt = os.path.join(folder_path, '{}.txt'.format(searcher.name))
		results = [searcher.search(p_text) for p_text in text_list]

		SearchExplainer(text_list, results).explain_save_txt(output_txt)


def run_test_all():
	pg = CJFHPatientGenerator()
	text_list = [p[0] for p in pg.get_text_patients()]
	folder_path = RESULT_PATH + '/text_handle/CJFH'
	test_all(text_list, folder_path)


def test_rules():
	text = """
无法并足站立
"""
	sg = UMLSSynGenerator()
	syn_dict, syn_dict_name = sg.get_hpo_to_source_syn_terms(), 'SourceSynDict'
	hpo_to_golds = dict_list_combine(neg_rules_to_dict(neg_rules), pos_rules_to_dict(pos_rules))
	hpo_to_gold_ps = dict_list_combine(neg_rules_to_dict(neg_prules), pos_rules_to_dict(pos_prules))

	matcher = BagTermMatcher(syn_dict, syn_dict_name, hpo_to_golds, hpo_to_gold_ps)
	searcher = MaxInvTextSearcher(matcher, invalid_end, search_skip)
	search_results = [searcher.search(text)]
	print(SearchExplainer([text], search_results).explain_as_str())


if __name__ == '__main__':
	pass

	hpo_source_umls_rules()



