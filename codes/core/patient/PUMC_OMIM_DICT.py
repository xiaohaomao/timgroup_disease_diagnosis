

CLASS_TO_OMIM = {
	'Castleman病': [ # 在名录中（16. Castleman 病），有映射（1）
		'OMIM:148000', # KAPOSI SARCOMA, SUSCEPTIBILITY TO; MULTICENTRIC CASTLEMAN DISEASE, SUSCEPTIBILITY TO, INCLUDED; MCD, INCLUDED; 卡波西肉瘤
	],
	'库欣综合征': [  # 不在名录中；搜索"cushing"
		'OMIM:219090', # PITA4

		'OMIM:219080', # AIMAH1
		'OMIM:615954', # AIMAH2

		'OMIM:610489', # PPNAD1
		'OMIM:610475', # PPNAD2
		'OMIM:614190', # PPNAD3
		'OMIM:615830', # PPNAD4
	],
	'特发性肺动脉高压': [ # 在名录中（54. 特发性肺动脉高压），但无映射；搜索"PRIMARY PULMONARY HYPERTENSION"
		'OMIM:178600', # PPH1
		'OMIM:615342', # PPH2
		'OMIM:615343', # PPH3
		'OMIM:615344', # PPH4
		'OMIM:265400', # PULMONARY HYPERTENSION, PRIMARY, AUTOSOMAL RECESSIVE
		# 'OMIM:268500', # ROWLEY-ROSENBERG SYNDROME; GROWTH RETARDATION, PULMONARY HYPERTENSION, AND AMINO ACIDURIA
	],
	'扩张型心肌病': [ # 名录中（52. 特发性心肌病）的子类，但无映射；搜索"dilated cardiomyopathy"
		'OMIM:115200', # CMD1A; CARDIOMYOPATHY, DILATED, 1A; CARDIOMYOPATHY, IDIOPATHIC DILATED
		'OMIM:600884', # CMD1B
		'OMIM:612877', # CMD1BB
		'OMIM:601493', # CMD1C
		'OMIM:613122', # CMD1CC
		'OMIM:601494', # CMD1D
		'OMIM:613172', # CMD1DD
		'OMIM:601154', # CMD1E
		'OMIM:613252', # CMD1EE
		'OMIM:601419', # CMD1F
		'OMIM:613286', # CMD1FF
		'OMIM:604145', # CMD1G
		'OMIM:613642', # CMD1GG
		'OMIM:604288', # CMD1H
		'OMIM:613881', # CMD1HH
		'OMIM:604765', # CMD1I
		'OMIM:615184', # CMD1II
		'OMIM:605362', # CMD1J
		'OMIM:615235', # CMD1JJ
		'OMIM:605582', # CMD1K
		'OMIM:615248', # CMD1KK
		'OMIM:606685', # CMD1L
		'OMIM:615373', # CMD1LL
		'OMIM:607482', # CMD1M
		'OMIM:615396', # CMD1MM
		'OMIM:615916', # CMD1NN
		'OMIM:608569', # CMD1O
		'OMIM:609909', # CMD1P
		'OMIM:609915', # CMD1Q
		'OMIM:613424', # CMD1R
		'OMIM:613426', # CMD1S
		# No CMD1T
		'OMIM:613694', # CMD1U
		'OMIM:613697', # CMD1V
		'OMIM:611407', # CMD1W
		'OMIM:611615', # CMD1X
		'OMIM:611878', # CMD1Y
		'OMIM:611879', # CMD1Z
		'OMIM:611880', # CMD2A
		'OMIM:614672', # CMD2B
		'OMIM:618189', # CMD2C
		'OMIM:302045', # CMD3B

		'OMIM:605676', # DCWHK; CARDIOMYOPATHY, DILATED, WITH WOOLLY HAIR AND KERATODERMA
		'OMIM:615821', # DCWHKTA; CARDIOMYOPATHY, DILATED, WITH WOOLLY HAIR, KERATODERMA, AND TOOTH AGENESIS
		'OMIM:610198', # MGCA5; CARDIOMYOPATHY, DILATED, WITH ATAXIA
		'OMIM:616117', # CCDD; CARDIAC CONDUCTION DISEASE WITH OR WITHOUT DILATED CARDIOMYOPATHY; 心脏传导疾病, 伴或不伴扩张型心肌病
	],

	'肥厚型心肌病':[
		'OMIM:613873',
		'OMIM:612124',
		'OMIM:613765',
		'OMIM:613255',
		'OMIM:608758',
		'OMIM:613874',
		'OMIM:612098',
		'OMIM:192600',
		'OMIM:115197',
		'OMIM:613876',
		'OMIM:115196',
		'OMIM:613690',
		'OMIM:607487',
		'OMIM:613875',
		'OMIM:115195',
		'OMIM:613838',
		'OMIM:613243',
		'OMIM:608751',
		'OMIM:613251',
		'OMIM:613873',
	],

	'Sack-Barabas综合征':[
		'OMIM:611816',


	],

	'Churg-Strauss综合征': [], # 不在名录中；不在OMIM中
	'特纳综合征': [], # 不在名录中；不在OMIM中
	'皮肤恶性黑色素瘤': [ # 不在名录中，此处指皮肤恶性黑色素瘤，搜索"MELANOMA, CUTANEOUS MALIGNANT"
		'OMIM:155600', # CMM1
		'OMIM:155601', # CMM2
		'OMIM:609048', # CMM3
		'OMIM:608035', # CMM4
		'OMIM:613099', # CMM5
		'OMIM:613972', # CMM6
		'OMIM:612263', # CMM7
		'OMIM:614456', # CMM8
		'OMIM:615134', # CMM9
		'OMIM:615848', # CMM10

		'OMIM:601800', # SHEP3; CMM8 INCLUDED
		'OMIM:606719', # MELANOMA-PANCREATIC CANCER SYNDROME
	],
	'抗NMDAR受体脑炎': [], # 在名录中（9. 自身免疫性脑炎），但无映射；不在OMIM中
	'Cronkhite-Canada综合征': [ # 不在名录中；搜索"Cronkhite-Canada"
		'OMIM:175500', # CRONKHITE-CANADA SYNDROME
	],
	'SAPHO': [ # 不在名录中；不在OMIM中
		# 'OMIM:612852', # 这个挺像的: OSTEOMYELITIS, STERILE MULTIFOCAL, WITH PERIOSTITIS AND PUSTULOSIS; OMPP
	],
	'重症肌无力': [ # 在名录中（32. 全身型重症肌无力），有映射（3）
		'OMIM:254200', # MG
		#'OMIM:607085', # MYAS1
		#'OMIM:159400', # MYASTHENIA GRAVIS, LIMB-GIRDLE
	],
	'POEMS综合症': [], # 在名录中（91．POEMS 综合征），无映射；不在OMIM中
	'先天性脊柱侧凸': [ # 在名录中（23. 先天性脊柱侧凸），无映射；搜索"SCOLIOSIS, CONGENITAL"

		# 名录"脊柱肋骨发育不全"
		'OMIM:277300', # SCDO1; SPONDYLOCOSTAL DYSOSTOSIS 1, AUTOSOMAL RECESSIVE; 常染色体隐性遗传的脊椎肋骨发育不全1
		'OMIM:608681', # SCDO2; SPONDYLOCOSTAL DYSOSTOSIS 2, AUTOSOMAL RECESSIVE;
		'OMIM:609813', # SCDO3; SPONDYLOCOSTAL DYSOSTOSIS 3, AUTOSOMAL RECESSIVE;
		'OMIM:613686', # SCDO4; SPONDYLOCOSTAL DYSOSTOSIS 4, AUTOSOMAL RECESSIVE;
		'OMIM:122600', # SCDO5; SPONDYLOCOSTAL DYSOSTOSIS 5; SCOLIOSIS, CONGENITAL, WITH OR WITHOUT RIB ANOMALIES TACS; 先天性脊柱侧凸（有或无肋骨异常）
		'OMIM:616566', # SCDO6; SPONDYLOCOSTAL DYSOSTOSIS 6, AUTOSOMAL RECESSIVE;

		# 名录"KLIPPEL-FEIL综合征"
		'OMIM:118100', # KFS1; KLIPPEL-FEIL SYNDROME 1, AUTOSOMAL DOMINANT;
		'OMIM:214300', # KFS2;
		'OMIM:613702', # KFS3
		'OMIM:616549', # KFS4

		# 名录"Alagille 综合征"
		'OMIM:118450', # ALGS1; ALAGILLE SYNDROME 1
		'OMIM:610205', # ALGS2

		# 名录"VACTERL 综合征"
		'OMIM:192350', # VATER/VACTERL ASSOCIATION

		# 其他
		'OMIM:272460', # SCT; SPONDYLOCARPOTARSAL SYNOSTOSIS SYNDROME; SCOLIOSIS, CONGENITAL, WITH UNILATERAL UNSEGMENTED BAR; 先天性脊柱侧凸（单侧不完整BAR）
		'OMIM:618578', # MYOSCO; MYOPATHY, CONGENITAL, PROGRESSIVE, WITH SCOLIOSIS; 先天性肌病伴脊柱侧凸
		'OMIM:255995', # MYPBB; MYOPATHY, CONGENITAL, BAILEY-BLOCH; MYOPATHY, CONGENITAL, WITH MYOPATHIC FACIES, SCOLIOSIS, AND MALIGNANT HYPERTHERMIA; 先天性肌病伴肌病相、脊柱侧凸、恶性高热

		# 由于名录中特发性脊柱侧凸与先天性脊柱侧凸属于鉴别诊断，故删去
		# 'OMIM:181800', # IS1; AIS; SCOLIOSIS, ISOLATED, SUSCEPTIBILITY TO, 1; 青少年型特发性脊柱侧凸
		# 'OMIM:607354', # IS2
		# 'OMIM:608765', # IS3
		# 'OMIM:612238', # IS4
		# 'OMIM:612239', # IS5
	],
	'烟雾病': [ # 不在名录中；搜索"MOYAMOYA"
		'OMIM:252350', # MYMY1
		'OMIM:607151', # MYMY2
		'OMIM:608796', # MYMY3
		'OMIM:300845', # MYMY4
		'OMIM:614042', # MYMY5
		'OMIM:615750', # MYMY6
	],
	'IgG4相关性疾病': [ # 在名录中（56. IgG4 相关性疾病）；有映射（1）
		'OMIM:228800', # FIBROSCLEROSIS, MULTIFOCAL; 多灶型纤维硬化
	],
	'21-羟化酶缺乏': [ # 在名录中（1. 21-羟化酶缺乏症）；有映射（1）;
		'OMIM:201910',
	],
	'原发性系统性淀粉样变性': [ # 在名录中（96. 原发性轻链型淀粉样变），为原发性系统性淀粉样变性的一种；
		'OMIM:254500', # MYELOMA, MULTIPLE; AMYLOIDOSIS, SYSTEMIC, INCLUDED; 映射来自ORPHA:314701; 泛化描述

		# Description中说，历史上，系统性淀粉样变被分为：遗传性、AL型（以前被称为原发性淀粉样变，由单克隆免疫球蛋白Ig轻链导致）、AA型（反应性）
		# 'OMIM:105200', # AMYLOIDOSIS, FAMILIAL VISCERAL; AMYLOIDOSIS, SYSTEMIC NONNEUROPATHIC; 家族性内脏淀粉样变, 主要涉及肾脏; 不确定
		# 'OMIM:105210', # FAP; AMYLOIDOSIS, HEREDITARY, TRANSTHYRETIN-RELATED; 家族性淀粉样多发性神经病，与甲状腺激素相关; 不确定
		# 'OMIM:105120', # AMYLOIDOSIS, FINNISH TYPE;
	],
	'直肠神经内分泌瘤': [ # 不在名录中；搜索"类癌"、"神经内分泌瘤"
		'OMIM:114900', # CARCINOID TUMORS, INTESTINAL
	],
	'Gitelman': [ # 在名录中（33. Gitelman 综合征）；有映射（1）
		'OMIM:263800', # GITELMAN SYNDROME; GTLMNS
	],
	'Fabry病': [ # 在名录中（27. 法布雷病），有映射（1）
		'OMIM:301500',
	],
	'Bartter综合征': [ # 不在名录中；搜索"BARTTER"
		'OMIM:601678', # BARTS1
		'OMIM:241200', # BARTS2
		'OMIM:607364', # BARTS3
		'OMIM:602522', # BARTS4A
		'OMIM:613090', # BARTS4B
		'OMIM:300971', # BARTS5

		'OMIM:601198', # HYPOC1; HYPOCALCEMIA, AUTOSOMAL DOMINANT 1, WITH BARTTER SYNDROME, INCLUDED
	],
	'蕈样肉芽肿': [ # 不在名录中；搜索"Mycosis fungoides"
		'OMIM:254400', # MYCOSIS FUNGOIDES
	],
	'Brugada综合征': [ # 在名录中（14. 心脏离子通道病），无映射；搜索"Brugada"
		'OMIM:601144', # BRGDA1
		'OMIM:611777', # BRGDA2
		'OMIM:611875', # BRGDA3
		'OMIM:611876', # BRGDA4
		'OMIM:612838', # BRGDA5
		'OMIM:613119', # BRGDA6
		'OMIM:613120', # BRGDA7
		'OMIM:613123', # BRGDA8
		'OMIM:616399', # BRGDA9
	],
	'朗格汉斯细胞组织细胞增生症': [ # 在名录中（60. 朗格汉斯细胞组织细胞增生症），有映射（2）
		'OMIM:604856', # LANGERHANS CELL HISTIOCYTOSIS
		'OMIM:246400', # LETTERER-SIWE DISEASE

		# 其他组织细胞增生
		# 'OMIM:602782', # SHML; HISTIOCYTOSIS-LYMPHADENOPATHY PLUS SYNDROME; 组织细胞增生症淋巴结肿大综合征; 百度百科与LCH存在鉴别诊断
	],
	'限制性心肌病': [ # 在名录中（52. 特发性心肌病），但无映射；搜索"restrictive cardio-myopathy"
		'OMIM:115210', # RCM1
		'OMIM:609578', # RCM2
		'OMIM:612422', # RCM3
		'OMIM:615248', # CMD1KK; RCM4, INCLUDED
		'OMIM:617047', # CMH26; RCM5, INCLUDED
	],
	'肝豆状核变性': [ # 在名录中（37. 肝豆状核变性），有映射（1）
		'OMIM:277900', # WILSON DISEASE
	],
	'嗜酸性肉芽肿血管炎': [ # 不在名录中；搜索"eosinophilic granulomatosis with polyangiitis", 发现等于"Churg-Strauss综合征"
		# 'OMIM:608710', # GPA, 是否包括EGPA?
	],
	'肺泡蛋白沉积症': [ # 在名录中（100. 肺泡蛋白沉积症），有映射（1）
		'OMIM:610910', # PAP, ACQUIRED
		'OMIM:265120', # SMDP1; PAP CONGENITAL 1; 先天性PAP
		'OMIM:610913', # SMDP2; PAP CONGENITAL 2
		'OMIM:610921', # SMDP3; PAP CONGENITAL 3
		'OMIM:300770', # SMDP4; PAP CONGENITAL 4
		'OMIM:614370', # SMDP5; PAP 5

		'OMIM:618042', # PAPHG; PULMONARY ALVEOLAR PROTEINOSIS WITH HYPOGAMMAGLOBULINEMIA; 肺泡蛋白沉积症伴低丙种球蛋白血症
		'OMIM:615486', # ILLD; INTERSTITIAL LUNG AND LIVER DISEASE; PULMONARY ALVEOLAR PROTEINOSIS, REUNION ISLAND
	],
	'致心律失常性右室心肌病': [ # 在名录中（52. 特发性心肌病），但无映射; 搜索"arrhythmogenic right ventricular dysplasia/cardiomyopathy"
		'OMIM:107970', # ARVD1; ARVC1; ARRHYTHMOGENIC RIGHT VENTRICULAR DYSPLASIA, FAMILIAL, 1
		'OMIM:600996', # ARVD2; ARVC2
		'OMIM:602086', # ARVD3; ARVC3
		'OMIM:602087', # ARVD4; ARVC4
		'OMIM:604400', # ARVD5; ARVC5
		'OMIM:604401', # ARVD6; ARVC6
		'OMIM:601419', # MFM1; ARVD7; ARVC7
		'OMIM:607450', # ARVD8; ARVC8
		'OMIM:609040', # ARVD9; ARVC9
		'OMIM:610193', # ARVD10; ARVC10
		'OMIM:610476', # ARVD11; ARVC11
		'OMIM:611528', # ARVD12; ARVC12
		'OMIM:615616', # ARVD13; ARVC13

		'OMIM:601214', # NXD; CARDIOMYOPATHY, ARRHYTHMOGENIC RIGHT VENTRICULAR, WITH SKIN, HAIR, AND NAIL ABNORMALITIES;
	],
	'McCune-Albright综合征': [ # 在名录中（69. McCune-Albright 综合征），有映射（1）
		'OMIM:174800', # MAS
	],
	'心脏离子通道病（长QT间期综合征）': [ # 在名录中有（14. 心脏离子通道病），无映射；搜索"long QT syndrome"
		'OMIM:192500', # LQT1
		'OMIM:613688', # LQT2
		'OMIM:603830', # LQT3
		'OMIM:600919', # LQT4, INCLUDED
		'OMIM:613695', # LQT5
		'OMIM:613693', # LQT6
		'OMIM:170390', # LQT7
		'OMIM:618447', # LQT8
		'OMIM:611818', # LQT9
		'OMIM:611819', # LQT10
		'OMIM:611820', # LQT11
		'OMIM:612955', # LQT12
		'OMIM:613485', # LQT13
		'OMIM:616247', # LQT14
		'OMIM:616249', # LQT15

		'OMIM:601005', # TS; LONG QT SYNDROME WITH SYNDACTYLY;
		'OMIM:220400', # JLNS1; JERVELL AND LANGE-NIELSEN SYNDROME; DEAFNESS, CONGENITAL, AND FUNCTIONAL HEART DISEASE; PROLONGED QT INTERVAL IN EKG AND SUDDEN DEATH; 来自ORPHA:90647
		'OMIM:612347', # JLNS2; 来自ORPHA:90647; 长QT伴耳聋
	],
	'Ehlers-Danlos综合征': [ # 不在名录中；搜索"EHLERS-DANLOS"
		'OMIM:130000', # EDSCL1; EDS, CLASSIC TYPE, 1; 典型
		'OMIM:130010', # EDSCL2;
		'OMIM:606408', # EDSCLL; EDS, CLASSIC-LIKE
		'OMIM:618000', # EDSCLL2

		'OMIM:225400', # EDSKSCL1；EDS，KYPHOSCOLIOTIC TYPE 1; EDS6; EDS6A; 眼-脊柱侧凸型
		'OMIM:614557', # EDSKSCL2

		'OMIM:130070', # EDSSPD1; EDS, SPONDYLODYSPLASTIC TYPE; 脊柱弯曲成形型
		'OMIM:615349', # EDSSPD2
		'OMIM:612350', # EDSSPD3

		'OMIM:130080', # EDSPD1; EDS, PERIODONTAL TYPE, 1; EDS8; 牙周病型
		'OMIM:617174', # EDSPD2

		'OMIM:130060', # EDSARTH1; EDS, RTHROCHALASIA TYPE; EDS7A; 关节松弛型
		'OMIM:617821', # EDSARTH2;

		'OMIM:601776', # EDSMC1; EDS, MUSCULOCONTRACTURAL TYPE, 1; 肌肉挛缩型
		'OMIM:615539', # EDSMC2

		'OMIM:616471', # BTHLM2; EDSMYP; EDS, MYOPATHIC TYPE; 肌病型
		'OMIM:225320', # EDSCV; EDS, CARDIAC VALVULAR TYPE; 心瓣膜型

		'OMIM:130090', # EDS, UNSPECIFIED TYPE
		'OMIM:608763', # EDS, BEASLEY-COHEN TYPE
		'OMIM:130020', # EDSHMB; EDS, HYPERMOBILITY TYPE; EDS3; 高血脂型
		'OMIM:225410', # EDSDERMS; EDS, DERMATOSPARAXIS TYPE; 皮肤轴型
		'OMIM:130050', # EDSVASC; EDS, VASCULAR TYPE; EDS4; 血管型
		'OMIM:225310', # PLATELET DYSFUNCTION FROM FIBRONECTIN ABNORMALITY; EDS, DYSFIBRONECTINEMIC TYPE; EDS10; 纤连蛋白异常致血小板功能障碍

		'OMIM:314400', # CVD1; EDS5, FORMERLY
		'OMIM:229200', # BCS1; EDS6B, FORMERLY
		'OMIM:304150', # OHS; EDS, OCCIPITAL HORN TYPE, FORMERLY
		'OMIM:147900', # EHLERS-DANLOS SYNDROME, TYPE XI, FORMERLY
	],
	'结节性硬化症': [ # 在名录中（114. 结节性硬化症），有映射（2）
		'OMIM:191100', # TSC1
		'OMIM:613254', # TSC2
	],
	'线粒体脑肌病': [ # 名录中有（72. 线粒体脑肌病），有映射（1），取名录中列举的常见线粒体脑肌病综合征
		'OMIM:300816', # COXPD6; COMBINED OXIDATIVE PHOSPHORYLATION DEFICIENCY 6; 线粒体脑肌病，X-连锁

		# Kearn–Sayre 综合征（KSS）；搜索"KEARNS-SAYRE"
		'OMIM:530000', # KSS; KEARNS-SAYRE SYNDROME;

		# 慢性进行性眼外肌麻痹（CPEO）；搜索"Chronic Progressive External Ophthal"
		'OMIM:157640', # PEOA1; 进行性眼肌麻痹伴线粒体DNA缺失，常染色体显性遗传
		'OMIM:609283', # PEOA2
		'OMIM:609286', # PEOA3
		'OMIM:610131', # PEOA4
		'OMIM:613077', # PEOA5
		'OMIM:615156', # PEOA6
		'OMIM:258450', # PEOB1; 进行性眼肌麻痹伴线粒体DNA缺失，常染色体隐性遗传
		'OMIM:616479', # PEOB2
		'OMIM:617069', # PEOB3
		'OMIM:617070', # PEOB4
		'OMIM:618098', # PEOB5

		# 线粒体神经胃肠脑肌病（MINGIE）；搜索"MITOCHONDRIAL NEUROGASTROINTESTINAL ENCEPHALOPATHY"
		'OMIM:603041', # MTDPS1; MITOCHONDRIAL DNA DEPLETION SYNDROME 1 (MNGIE TYPE)
		'OMIM:613662', # MTDPS4B; MITOCHONDRIAL DNA DEPLETION SYNDROME 4B (MNGIE TYPE)
		'OMIM:612075', # MTDPS8A; MNGIE, RRM2B-RELATED, INCLUDED

		# 线粒体脑肌病伴卒中样发作和乳酸酸中毒（MELAS）; 搜索"MELAS"
		'OMIM:540000', # MELAS; MELAS SYNDROME


		# 肌阵挛伴破碎红纤维（MERRF）; 搜索"MERRF"
		'OMIM:545000', # MERRF; MERRF SYNDROME

		# 与线粒体DNA相关的Leigh综合征; 搜索"LEIGH"
		'OMIM:256000', # LS; LEIGH SYNDROME; LEIGH SYNDROME DUE TO MITOCHONDRIAL COMPLEX I/II/III/IV/V DEFICIENCY, INCLUDED

		# Pearson综合征; 搜索"PEARSON SYNDROME"
		'OMIM:557000', # PEARSON MARROW-PANCREAS SYNDROME

		# 神经原性肌无力，共济失调，视网膜色素变性（NARP）; 搜索"Neurogenic weakness, Ataxia, Retinitis Pigmentosa, NARP"
		'OMIM:551500', # NARP SYNDROME; NEUROPATHY, ATAXIA, AND RETINITIS PIGMENTOSA

		# Alpers-Huttenlocher综合征; 搜索"Alpers-Huttenlocher"
		'OMIM:203700', # MTDPS4A; ALPERS-HUTTENLOCHER SYNDROME; MITOCHONDRIAL DNA DEPLETION SYNDROME 4A (ALPERS TYPE);
	],
	'线粒体DNA缺失综合征': [ # 搜索"MITOCHONDRIAL DNA DEPLETION SYNDROME"
		'OMIM:603041', # MTDPS1; MITOCHONDRIAL DNA DEPLETION SYNDROME 1 (MNGIE TYPE)
		'OMIM:609560', # MTDPS2; MITOCHONDRIAL DNA DEPLETION SYNDROME 2 (MYOPATHIC TYPE)
		'OMIM:251880', # MTDPS3; MITOCHONDRIAL DNA DEPLETION SYNDROME 3 (HEPATOCEREBRAL TYPE)
		'OMIM:203700', # MTDPS4A; MITOCHONDRIAL DNA DEPLETION SYNDROME 4A (ALPERS TYPE)
		'OMIM:613662', # MTDPS4B; MITOCHONDRIAL DNA DEPLETION SYNDROME 4B (MNGIE TYPE)
		'OMIM:612073', # MTDPS5; MITOCHONDRIAL DNA DEPLETION SYNDROME 5 (ENCEPHALOMYOPATHIC WITH OR WITHOUT METHYLMALONIC ACIDURIA)
		'OMIM:256810', # MTDPS6; MITOCHONDRIAL DNA DEPLETION SYNDROME 6 (HEPATOCEREBRAL TYPE)
		'OMIM:271245', # MTDPS7; MITOCHONDRIAL DNA DEPLETION SYNDROME 7 (HEPATOCEREBRAL TYPE)
		'OMIM:612075', # MTDPS8A; MNGIE, RRM2B-RELATED, INCLUDED
		'OMIM:245400', # MTDPS9; MITOCHONDRIAL DNA DEPLETION SYNDROME 9 (ENCEPHALOMYOPATHIC TYPE WITH METHYLMALONIC ACIDURIA)
		'OMIM:212350', # MTDPS10; MITOCHONDRIAL DNA DEPLETION SYNDROME 10 (CARDIOMYOPATHIC TYPE)
		'OMIM:615084', # MTDPS11; MITOCHONDRIAL DNA DEPLETION SYNDROME 11
		'OMIM:617184', # MTDPS12A; MITOCHONDRIAL DNA DEPLETION SYNDROME 12A (CARDIOMYOPATHIC TYPE)
		'OMIM:615418', # MTDPS12B; MITOCHONDRIAL DNA DEPLETION SYNDROME 12B (CARDIOMYOPATHIC TYPE)
		'OMIM:615471', # MTDPS13; MITOCHONDRIAL DNA DEPLETION SYNDROME 13 (ENCEPHALOMYOPATHIC TYPE)
		'OMIM:616896', # MTDPS14; MITOCHONDRIAL DNA DEPLETION SYNDROME 14 (CARDIOENCEPHALOMYOPATHIC TYPE)
		'OMIM:617156', # MTDPS15; MITOCHONDRIAL DNA DEPLETION SYNDROME 15 (HEPATOCEREBRAL TYPE)
		'OMIM:618528', # MTDPS16; MITOCHONDRIAL DNA DEPLETION SYNDROME 16 (HEPATIC TYPE)
		'OMIM:618567', # MTDPS17; MITOCHONDRIAL DNA DEPLETION SYNDROME 17
	],
	'Fanconi综合征': [ # 搜索"FANCONI RENOTUBULAR SYNDROME"；不确定
		'OMIM:134600', # FRTS1; FANCONI RENOTUBULAR SYNDROME 1
		'OMIM:613388', # FRTS2
		'OMIM:615605', # FRTS3
		'OMIM:616026', # FRTS4
	],
	'CREST综合征': [ # 搜索"CREST"
		'OMIM:181750', # SCLERODERMA, FAMILIAL PROGRESSIVE; CREST SYNDROME, INCLUDED
	],
	'系统性硬化症': [ # 名录中有（112. 系统性硬化症）；搜索"systemic sclerosis"
		'OMIM:181750', # SCLERODERMA, FAMILIAL PROGRESSIVE
	],
	'着色性干皮病': [ # 搜索"Xeroderma pigmentosum"
		'OMIM:278700', # XPA; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP A
		'OMIM:610651', # XPB; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP B;
		'OMIM:278720', # XPC; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP C
		'OMIM:278730', # XPD; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP D
		'OMIM:278740', # XPE; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP E
		'OMIM:278760', # XPF; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP F
		'OMIM:278780', # XPG; XERODERMA PIGMENTOSUM, COMPLEMENTATION GROUP G
		'OMIM:278750', # XPV; XERODERMA PIGMENTOSUM, VARIANT TYPE;
		'OMIM:194400', # XERODERMA PIGMENTOSUM, AUTOSOMAL DOMINANT, MILD
	],
	# pumc_pk
	'Prader-Willi综合征': [ # 搜索"Prader-Willi"
		'OMIM:176270', # PWS; PRADER-WILLI SYNDROME
	],
	'马方综合征': [ # 搜索"marfan"
		'OMIM:154700', # MFS1; MARFAN SYNDROME; MARFAN SYNDROME, TYPE I; MFS1
		#'OMIM:610168', # LOEYS-DIETZ SYNDROME 2; LDS2; MARFAN SYNDROME, TYPE II, FORMERLY
	],
	'肌萎缩侧索硬化':[ # 搜索"amyotrophic lateral sclerosis"
		'OMIM:105400', # ALS1;
		'OMIM:205100', # ALS2; AMYOTROPHIC LATERAL SCLEROSIS 2, JUVENILE
		'OMIM:606640', # ALS3
		'OMIM:602433', # ALS4
		'OMIM:602099', # ALS5
		'OMIM:608030', # ALS6
		'OMIM:608031', # ALS7
		'OMIM:608627', # ALS8
		'OMIM:611895', # ALS9
		'OMIM:612069', # ALS10
		'OMIM:612577', # ALS11
		'OMIM:613435', # ALS12
		'OMIM:183090', # SCA2; ALS13, INCLUDED
		'OMIM:613954', # ALS14
		'OMIM:300857', # ALS15
		'OMIM:614373', # ALS16
		'OMIM:614696', # ALS17
		'OMIM:614808', # ALS18
		'OMIM:615515', # ALS19
		'OMIM:615426', # ALS20
		'OMIM:606070', # ALS21
		'OMIM:616208', # ALS22
		'OMIM:617839', # ALS23
		'OMIM:617892', # ALS24
		'OMIM:617921', # ALS25

		# 'OMIM:105550', # FTDALS1; FRONTOTEMPORAL DEMENTIA AND/OR AMYOTROPHIC LATERAL SCLEROSIS 1
		# 'OMIM:105500', # AMYOTROPHIC LATERAL SCLEROSIS-PARKINSONISM/DEMENTIA COMPLEX 1
	],
	'多系统萎缩':[
		'OMIM:146500', # MULTIPLE SYSTEM ATROPHY 1, SUSCEPTIBILITY TO; MSA1
	],
	'Alport综合征':[
		'OMIM:301050', # ATS1; ALPORT SYNDROME 1, X-LINKED;
		'OMIM:203780', # ATS2
		'OMIM:104200', # ATS3

		# 'OMIM:308940', # LEIOMYOMATOSIS, DIFFUSE, WITH ALPORT SYNDROME; DL-ATS
		# 'OMIM:155100', # ALPORT SYNDROME WITH MACROTHROMBOCYTOPENIA, FORMERLY; APSM,
		# 'OMIM:300194', # ALPORT SYNDROME, MENTAL RETARDATION, MIDFACE HYPOPLASIA, AND ELLIPTOCYTOSIS;
	],
	'阵发性睡眠性血红蛋白尿':[
		'OMIM:300818', # PAROXYSMAL NOCTURNAL HEMOGLOBINURIA 1; PNH1
		'OMIM:615399', # PAROXYSMAL NOCTURNAL HEMOGLOBINURIA 2; PNH2
	],
	'尼曼匹克病': [
		'OMIM:257200', # NIEMANN-PICK DISEASE, TYPE A
		'OMIM:607616', # NIEMANN-PICK DISEASE, TYPE B
		'OMIM:257220', # NIEMANN-PICK DISEASE, TYPE C1; NPC1
		'OMIM:607625', # NIEMANN-PICK DISEASE, TYPE C2; NPC2
	],
	### todo 2020 11 27 增加协和MDT数据的诊断测试 ###

	'阴道闭锁':[
		'OMIM:192050',
	],
	'家族性高胆固醇血症': [
		'OMIM:143890',     # 	家族性高胆固醇血症（FH）
		'OMIM:144010',     # 常染色体显性遗传型高胆固醇血症，B型
	],
	# CRESR综合征 上面已经有
	'纯合子家族性高胆固醇血症':[
		'OMIM:143890',
		'OMIM:144010',
	],
	'PROTEASOME-ASSOCIATED AUTOINFLAMMATORY SYNDROME 2':[
		'OMIM:618048',
		'OMIM:617591',
	],
	'Activated PI3K-delta syndrome':[],
	'Primary angiitis of the central nervous system':[],
	'粘多糖病第Ⅱ型':[
		'OMIM:309900',
	],
	'戈谢病':[
		'OMIM:230900',
		'OMIM:230800',
		'OMIM:608013',
		'OMIM:231000',
		'OMIM:231005',
		'OMIM:610539',
	],
	'脊髓小脑性共济失调': [
		'OMIM:603516',  #脊髓小脑性共济失调第10型
		'OMIM:604326',  #脊髓小脑性共济失调第12型
		'OMIM:605361',  #脊髓小脑性共济失调第14型
		'OMIM:609307',  #脊髓小脑性共济失调第27型
		'OMIM:604432',  #脊髓小脑性共济失调第11型
		'OMIM:605259',  #脊髓小脑性共济失调第13型
		'OMIM:607454',  #脊髓小脑性共济失调第21型
		'OMIM:117210',  #脊髓小脑共济失调31
		'OMIM:608768',  #脊髓小脑性共济失调第8型
		'OMIM:600224',  #脊髓小脑性共济失调第5型
		'OMIM:607346',  #脊髓小脑性共济失调第19/22型
		'OMIM:609306',  #脊髓小脑共济失调26（SCA26）
		'OMIM:117360',  #脊髓小脑共济失调29；SCA29
		'OMIM:614153',  #脊髓小脑共济失调36；SCA36
		'OMIM:606658',  #脊髓小脑性共济失调第15/16型
		'OMIM:610246',  #脊髓小脑性共济失调28（SCA28）
		'OMIM:613908',  #脊髓小脑性共济失调35; SCA35
		'OMIM:616053',  #脊髓小脑性共济失调40; SCA40
		'OMIM:615957',  #脊髓小脑性共济失调38（SCA38）
		'OMIM:616410',  #脊髓小脑性共济失调41（SCA41）
		#'OMIM:607136',  #脊髓小脑运动失调17型
		'OMIM:610245',  #脊髓小脑的共济失调23; SCA23
		#'OMIM:164400',  #脊髓小脑运动失调1型
		#'OMIM:183086',  #脊髓小脑运动失调6型
		#'OMIM:183090',  #脊髓小脑运动失调2型
		'OMIM:600223',  #脊髓小脑性共济失调第4型
		'OMIM:607458',  #脊髓小脑性共济失调第18型
		#'OMIM:133190',  #红斑角皮病伴共济失调(Spinocerebellar ataxia type 34) ?? 暂时不知道这个要不要加上去
		#'OMIM:164500',  #常染色体显性脊髓小脑运动失调7型
		#'OMIM:109150',  #脊髓小脑运动失调3型
		#'OMIM:183090',  #
	],
	'Nodular fasciitis': [
		'OMIM:226350',   #！*！
	],
	'高安动脉炎,无脉病,大动脉炎':[
		'OMIM:207600',
	],
	'Aicardi Goutieres综合征': [
		'OMIM:225750',
		'OMIM:610181',
		'OMIM:610329',
		'OMIM:615010',
		'OMIM:610333',
		'OMIM:615846',
		'OMIM:612952',
	],
	'缺铁性红细胞贫血伴b细胞免疫缺乏，周期性发烧和发育迟缓':[
		'OMIM:616084',
	],
	#Castleman病 已经有了
	#
	'先天性脂肪瘤样增生，血管畸形和表皮痣综合征': [
		'OMIM:612918',
	],
	'先天性多发关节挛缩综合征': [
		'OMIM:208100',
		'OMIM:617468',
		'OMIM:618484',
		#'OMIM:618766'
		#'OMIM:618947'
	],
	'macrodactyly':[
		'OMIM:155500',  #！*！
	],
	#McCune-Albright综合征 已经有了
	'肾上腺皮质癌':[
		'OMIM:202300',
	],
	'Oncogenic osteomalacia':[
		'OMIM:109130',   #！*！
	],
	# 肌萎缩侧索硬化 上面已经有了
	# Fabry病 上面已经有了
	'红细胞生成性原卟啉病':[
		'OMIM:177000',
	],
	'瓦登伯格综合征':[
		'OMIM:148820',   #克-瓦二氏综合征
		'OMIM:193500',   #瓦登伯格综合征I型
		'OMIM:193510',   #瓦登伯格综合征II型
		'OMIM:613265',   #瓦登伯革氏综合症4B型
		'OMIM:613266',   #瓦登伯革氏综合症4C型；WS4C
		'OMIM:277580',   #瓦登伯革氏综合症4A型（WS4A)
		'OMIM:606662',   #WAARDENBURG SYNDROME, TYPE 2C; WS2C
		'OMIM:600193',   #WAARDENBURG SYNDROME, TYPE 2B; WS2B
		'OMIM:611584',   #瓦登伯革氏症候群2E型（ WS2E）
		'OMIM:608890',   #2D型瓦登伯革氏症候群（WS2D）
	],
	# SAPHO 上面已经有了
	# APDS，Activated PI3K-delta syndrome 上面已经有了
	# 结节性硬化症 上面已经有了
	'Primary hypertrophic osteoarthropathy': [
		'OMIM:167100',   # HYPERTROPHIC OSTEOARTHROPATHY, PRIMARY, AUTOSOMAL DOMINANT; PHOAD
		'OMIM:614441',   # 肥大性骨关节病，原发性，常染色体隐性遗传，2型; PHPAR2
	],
	'脊髓性肌萎缩症': [
		'OMIM:253300',    #婴儿脊髓性肌萎缩
		'OMIM:253400',    #青少年脊髓型肌萎缩
		'OMIM:158600',     #脊髓性肌萎缩（下肢受累，常染色体显性遗传）-1；SMALED1
		'OMIM:615290',     #脊髓性肌萎缩（下肢受累，常染色体显性遗传）-2；SMALED2
	],

	# 蕈样肉芽肿 上面已经有了

	# 2020 11 30 为了做 CCRD所有疾病的测试 #
	#'21-羟化酶缺乏症': ['21-羟化酶缺乏症'],
	'白化病': [
		'OMIM:300500',
		'OMIM:203290',
		'OMIM:606952',
		'OMIM:606574',
		'OMIM:203200',
		'OMIM:214500',
		'OMIM:300700',
		'OMIM:203100',
		'OMIM:615179',
		'OMIM:103470',

	],
	#'Alport 综合征': [''],
	#'肌萎缩侧索硬化': [''],
	'天使综合征': [
		'OMIM:105830',
	],
	'精氨酸酶缺乏症': [
		'OMIM:207800',
	],
	'窒息性胸腔失养症': [],
	'非典型溶血性尿毒症综合征': [],
	'自身免疫性脑炎': [],
	#'抗 NMDAR 脑炎': [''],
	'抗 LGI1 抗体相关脑炎': [],
	'自身免疫性垂体炎': [],
	'自身免疫性胰岛素受体病': [],
	'β-酮硫解酶缺乏症': [],
	'生物素酶缺乏症': [],
	#'长QT综合征': [''],
	'短QT综合征': [],
	#'Brugada综合征': [''],
	'儿茶酚胺敏感型多形性室性心动过速': [],
	'原发性肉碱缺乏症': [],
	#'Castleman 病': [''],
	'腓骨肌萎缩症': [],
	'瓜氨酸血症': [],
	'先天性肾上腺发育不良': [],
	'先天性高胰岛素性低血糖血症': [],
	'先天性肌无力综合征': [],
	'先天性肌强直': [],
	#'先天性脊柱侧凸': [''],
	'冠状动脉扩张': [],
	'先天性纯红细胞再生障碍性贫血': [],
	'Erdheim-Chester 病': [],
	#'法布雷病': [''],
	'家族性地中海热': [],
	'范可尼贫血': [],
	'半乳糖血症': [],
	#'戈谢病': [''],
	#'全身型重症肌无力': [''],
	#'Gitelman 综合征': [''],
	'戊二酸血症 I 型': [],
	'多种酰基辅酶 A 脱氢酶缺乏症': [],
	'糖原累积病Ia型': [],
	'糖原累积病Ib型': [],
	'糖原累积病II型': [],
	'血友病': [],
	#'肝豆状核变性': [''],
	'遗传性血管水肿': [],
	'单纯型大疱性表皮松解症': [],
	'交界型大疱性表皮松解症': [],
	'营养不良型大疱性表皮松解症': [],
	'遗传性果糖不耐受症': [],
	'遗传性低镁血症': [],
	'遗传性多发脑梗死性痴呆': [],
	'遗传性痉挛性截瘫': [],
	'全羧化酶合成酶缺乏症': [],
	'同型半胱氨酸血症': [],
	#'纯合子家族性高胆固醇血症': [''],
	'亨廷顿舞蹈病': [],
	'HHH 综合征': [],
	'高苯丙氨酸血症': [],
	'低磷酸酯酶症': [],
	'低血磷性佝偻病': [],
	#'家族性/特发性扩张型心肌病': [''],
	#'致心律失常性右室发育不良/心肌病': [''],
	#'家族性/特发性限制型心肌病': [''],
	'左室致密化不全': [],
	'遗传性转甲状腺素蛋白相关淀粉样变': [],
	'特发性低促性腺激素性性腺功能减退症': [],
	'卡尔曼综合征': [],
	#'特发性肺动脉高压': [],
	'特发性肺纤维化': [],
	#'IgG4 相关性疾病': [],
	'先天性胆汁酸合成障碍': [],
	'异戊酸血症': [],
	#'朗格汉斯细胞组织细胞增生症': [''],
	'莱伦综合征': [],
	'Leber 遗传性视神经病变': [],
	'长链 3 羟酰基辅酶 A 脱氢酶缺乏症': [],
	'淋巴管肌瘤病': [],
	'赖氨酸尿蛋白不耐受症': [],
	'溶酶体酸性脂肪酶缺乏症': [],
	'枫糖尿症': [],
	#'马方综合征': [''],
	#'McCune-Albright 综合征': [''],
	'中链酰基辅酶 A 脱氢酶缺乏症': [],
	'甲基丙二酸血症': [],
	#'线粒体脑肌病伴卒中样发作和乳酸酸中毒': [''],
	#'肌阵挛伴破碎红纤维': [''],
	#'Leigh 综合征': [''],
	#'慢性进行性眼外肌麻痹': [''],
	#'线粒体神经胃肠脑肌病': [''],
	#'Kearn–Sayre 综合征': [''],
	#'Pearson综合征': [''],
	#'神经原性肌无力，共济失调，视网膜色素变性': [''],
	#'Alpers-Huttenlocher综合征': [''],
	'黏多糖贮积症': [],
	'多灶性运动神经病': [],
	'多发性硬化': [],
	#'多系统萎缩)': [''],
	'强直性肌营养不良': [],
	'N-乙酰谷氨酸合成酶缺乏症': [],
	'新生儿糖尿病': [],
	'视神经脊髓炎': [],
	#'尼曼匹克病A型': [''],
	#'尼曼匹克病B型': [''],
	#'尼曼匹克病C型': [''],
	'非综合征性耳聋': [],
	'Noonan 综合征': [],
	'鸟氨酸氨甲酰胺基转移酶缺乏症': [],
	'成骨不全症': [],
	'帕金森病': [],
	#'阵发性睡眠性血红蛋白尿症': [''],
	'黑斑息肉综合征': [],
	'苯丙酮尿症': [],
	#'POEMS 综合征': [''],
	#'卟啉病': [''],
	#'Prader-Willi 综合征': [''],
	#'原发性联合免疫缺陷病': [''],
	'原发性遗传性肌张力不全': [],
	#'原发性轻链型淀粉样变': [''],
	'进行性家族性肝内胆汁淤积症': [],
	'进行性肌营养不良': [],
	'丙酸血症': [],
	#'肺泡蛋白沉积症': [''],
	'囊性纤维化': [],
	'视网膜色素变性': [],
	'视网膜母细胞瘤': [],
	'重症先天性粒细胞缺乏症': [],
	'婴儿严重肌阵挛性癫痫': [],
	'镰刀型细胞贫血病': [],
	'Silver－Russell  综合征': [],
	'谷固醇血症': [],
	'脊髓延髓肌萎缩症': [],
	#'脊髓性肌萎缩症': [''],
	#'脊髓小脑性共济失调': [''],
	#'系统性硬化症': [''],
	'四氢生物蝶呤缺乏症': [],
	#'结节性硬化症': [''],
	'酪氨酸血症I型': [],
	'极长链酰基辅酶 A 脱氢酶缺乏症': [],
	'威廉姆斯综合征': [],
	'湿疹血小板减少伴免疫缺陷综合征': [],
	'肾上腺脑白质营养不良': [],
	'X-连锁无丙种球蛋白血症': [],
	'X-连锁淋巴增生症':[],

}


if __name__ == '__main__':
	def no_strip(s):
		return len(s) == len(s.strip())
	def no_repeat(s):
		return len(set(s)) == len(s)
	def check():
		d = CLASS_TO_OMIM
		for diag_str, diag_codes in d.items():
			assert no_strip(diag_str)
			for diag_code in diag_codes:
				assert no_strip(diag_code)
			assert no_repeat(diag_codes)
	check()

