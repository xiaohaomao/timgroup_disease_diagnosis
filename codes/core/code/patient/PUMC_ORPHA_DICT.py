

CLASS_TO_ORPHANET = {
	'Castleman病': [ # 搜索"Castleman"
		'ORPHA:160', # Castleman disease (Disorder);
		'ORPHA:93685', # Localized Castleman disease (Subtype of disorder); 局限性
		'ORPHA:93686', # Multicentric Castleman disease (Subtype of disorder); 多中心
		'ORPHA:93682', # Pediatric Castleman disease (Subtype of disorder); 儿童
	],
	'Sack-Barabas综合征':[
		'ORPHA:420561',
	],
	'肥厚型心肌病': [],

	'库欣综合征':[ # 搜索"cushing"
		'ORPHA:553', # Cushing syndrome (Group of disorders)
		'ORPHA:96253', # Cushing disease; Pituitary-dependent Cushing syndrome (Disorder)
		'ORPHA:99892', # ACTH-dependent Cushing syndrome (Group of disorders)
		'ORPHA:99893', # ACTH-independent Cushing syndrome (Group of disorders)
		'ORPHA:443287', # ACTH-independent Cushing syndrome due to rare cortisol-producing adrenal tumor (Group of disorders)
		'ORPHA:99889', # Cushing syndrome due to ectopic ACTH secretion (Disorder)
		'ORPHA:189427', # Cushing syndrome due to macronodular adrenal hyperplasia (Disorder)
	],
	'特发性肺动脉高压': [ # 搜索"Idiopathic pulmonary arterial hypertension"
		'ORPHA:275766', # Idiopathic pulmonary arterial hypertension (Subtype of disorder)
		'ORPHA:422', # Idiopathic/heritable pulmonary arterial hypertension
	],
	'扩张型心肌病': [ # 搜索"dilated cardiomyopathy"
		'ORPHA:154', # Familial isolated dilated cardiomyopathy (Disorder)
		'ORPHA:217604', # Dilated cardiomyopathy (Group of disorders)
			'ORPHA:217607', # Familial dilated cardiomyopathy (Group of disorders)
				'ORPHA:217610', # Neuromuscular disease with dilated cardiomyopathy (Group of disorders)
				'ORPHA:217613', # Mitochondrial disease with dilated cardiomyopathy (Group of disorders)
				'ORPHA:217616', # Fatty acid oxidation and ketogenesis disorder with dilated cardiomyopathy (Group of disorders)
				'ORPHA:217619', # Syndrome associated with dilated cardiomyopathy (Group of disorders)
				'ORPHA:371176', # Congenital disorder of glycosylation with dilated cardiomyopathy (Group of disorders)
			'ORPHA:217629', # Non-familial dilated cardiomyopathy (Group of disorders)
				'ORPHA:324767', # Non-familial rare disease with dilated cardiomyopathy (Group of disorders)
		'ORPHA:65282', # Carvajal syndrome; Woolly hair-palmoplantar hyperkeratosis-dilated cardiomyopathy syndrome (Disorder); OMIM: 605676 | OMIM:615821
		'ORPHA:66634', # Dilated cardiomyopathy with ataxia (Disorder); OMIM:610198
	],
	'Churg-Strauss综合征': [ # 搜索"Churg"
		'ORPHA:183',
	],
	'特纳综合征': [ # 搜索"Turner"
		'ORPHA:881', # Turner syndrome (Disorder)
		'ORPHA:99413', # Turner syndrome due to structural X chromosome anomalies
	],
	'皮肤恶性黑色素瘤': [ # 搜索"melanoma"
		'ORPHA:618', # Familial melanoma; 与OMIM的CMM对应
		'ORPHA:404560', # Familial atypical multiple mole melanoma syndrome (Disorder); OMIM:155600 | OMIM:606719
	],
	'抗NMDAR受体脑炎': [ # 搜索"NMDA"
		'ORPHA:217253', # Limbic encephalitis with NMDA receptor antibodies (Disorder)
	],
	'Cronkhite-Canada综合征': [ # 搜索"Cronkhite"
		'ORPHA:2930', # Cronkhite-Canada syndrome
	],
	'SAPHO': [ # 搜索"SAPHO"
		'ORPHA:793', # SAPHO syndrome
	],
	'重症肌无力': [ # 搜索"Myasthenia gravis"
		'ORPHA:589', # Myasthenia gravis (Disorder); OMIM:254200
		'ORPHA:391490', # Adult-onset myasthenia gravis (Subtype of disorder); 成年型
		'ORPHA:391497', # Juvenile myasthenia gravis (Subtype of disorder); 青少年型
		'ORPHA:391504', # Transient neonatal myasthenia gravis (Subtype of disorder); 新生儿型
	],
	'POEMS综合症': [ # 搜索"POEMS"
		'ORPHA:2905', # POEMS syndrome (Disorder)
	],
	'先天性脊柱侧凸': [
		# 名录"脊柱肋骨发育不全"
		'ORPHA:2311', # OMIM:277300  608681  609813  613686  616566
		'ORPHA:1797', # OMIM:122600,

		# 名录"KLIPPEL-FEIL综合征"
		'ORPHA:2345', # OMIM:118100  214300  613702
		'ORPHA:447974', # OMIM:616549

		# 名录"Alagille 综合征
		'ORPHA:52', # OMIM:118450  610205

		# 名录"VACTERL 综合征"
		'ORPHA:887', # OMIM:192350

		# 其他
		'ORPHA:3275', # OMIM:272460
		'ORPHA:168572', # OMIM:255995
	],
	'烟雾病': [ # 搜索"MOYAMOYA"
		'ORPHA:477768', # Moyamoya angiopathy (Group of disorders)
		'ORPHA:2573', # Moyamoya disease (Disorder)
		'ORPHA:280679', # Moyamoya angiopathy-short stature-facial dysmorphism-hypergonadotropic hypogonadism syndrome (Disorder); OMIM:300845
		'ORPHA:401945', # Moyamoya disease with early-onset achalasia
	],
	'IgG4相关性疾病': [ # 搜索"IgG4-related"
		'ORPHA:284264', # IgG4-related disease (Group of disorders)
		'ORPHA:280302', # Autoimmune pancreatitis type 1;  IgG4-related pancreatitis (Subtype of disorder)
		'ORPHA:451607', # Cutaneous pseudolymphoma; gG4-related skin disease (Disorder)
		'ORPHA:449566', # Eosinophilic angiocentric fibrosis; IgG4-related eosinophilic angiocentric fibrosis (Disorder)
		'ORPHA:449400', # IgG4-related aortitis; IgG4-related periaortitis (Disorder)
		'ORPHA:79078', # IgG4-related dacryoadenitis and sialadenitis (Disorder)
		'ORPHA:449395', # IgG4-related kidney disease (Disorder)
		'ORPHA:63999', #  IgG4-related mediastinitis (Disorder)
		'ORPHA:238593', # IgG4-related mesenteritis (Disorder)
		'ORPHA:449563', # IgG4-related ophthalmic disease (Disorder)
		'ORPHA:449427', # IgG4-related pachymeningitis (Disorder)
		'ORPHA:49041', # IgG4-related retroperitoneal fibrosis (Disorder); OMIM:228800
		'ORPHA:447764', # IgG4-related sclerosing cholangitis (Disorder)
		'ORPHA:449432', # IgG4-related submandibular gland disease (Disorder)
		'ORPHA:64744', # IgG4-related thyroid disease (Disorder)
		'ORPHA:555437', #  Lymphoplasmacytic inflammatory pseudotumor of the liver; IgG4-related inflammatory pseudotumor of the liver (Subtype of disorder)
		'ORPHA:451602', # Primary cutaneous plasmacytosis (Disorder)
	],
	'21-羟化酶缺乏': [ # 搜索"21-hydroxylase deficiency"
		'ORPHA:90794', # Classic congenital adrenal hyperplasia due to 21-hydroxylase deficiency (Disorder); OMIM:201910
			'ORPHA:315306', # Classic congenital adrenal hyperplasia due to 21-hydroxylase deficiency, salt wasting form; Subtype of disorder
			'ORPHA:315311', # Classic congenital adrenal hyperplasia due to 21-hydroxylase deficiency, simple virilizing; Subtype of disorder
	],
	'原发性系统性淀粉样变性': [ # 搜索"Primary systemic amyloidosis"; "AL amyloidosis"
		'ORPHA:85443', # AL amyloidosis (Disorder); Definition中说
		'ORPHA:314701' # Primary systemic amyloidosis (Subtype of disorder)
		# 'ORPHA:85445', # AA amyloidosis; AA型
	],
	'直肠神经内分泌瘤': [ # 搜索"Carcinoid tumor"
		'ORPHA:100093', #  Carcinoid syndrome;  (Disorder);
	],
	'Gitelman': [ # 搜索"Gitelman"
		'ORPHA:358', # Gitelman syndrome (Disorder); OMIM:263800
	],
	'Fabry病': [ # 搜索"Fabry"
		'ORPHA:324', # Fabry disease (Disorder); OMIM:301500
	],
	'Bartter综合征': [ # 搜索"Bartter"
		'ORPHA:93604', # Antenatal Bartter syndrome; Bartter syndrome, furosemide type (Subtype of disorder); OMIM:601678
		'ORPHA:112', # Bartter syndrome (Disorder)
		'ORPHA:263417', # Bartter syndrome with hypocalcemia (Subtype of disorder); OMIM:601198
		'ORPHA:93605', # Classic Bartter syndrome (Subtype of disorder); OMIM:607364
		'ORPHA:89938', # Infantile Bartter syndrome with sensorineural deafness (Subtype of disorder); OMIM:602522 | OMIM:613090
	],
	'蕈样肉芽肿': [ # 搜索"Mycosis fungoides"
		'ORPHA:2584', # Classic mycosis fungoides (Disorder); OMIM:254400
		'ORPHA:178512', # Mycosis fungoides-associated follicular mucinosis (Disorder)
		'ORPHA:178566', # Mycosis fungoides and variants (Group of disorders)
	],
	'Brugada综合征': [ # 搜索"Brugada"
		'ORPHA:130', # Brugada syndrome (Disorder)
	],
	'朗格汉斯细胞组织细胞增生症': [ # 搜索"LANGERHANS"
		'ORPHA:389', # Langerhans cell histiocytosis (Disorder)
		'ORPHA:99870', # Letterer-Siwe disease
		'ORPHA:99871', # Eosinophilic granuloma; 嗜酸性肉芽肿
		'ORPHA:99872', # Hashimoto-Pritzker syndrome
		'ORPHA:99873', # Hand-Schüller-Christian disease
		'ORPHA:99874', # Adult pulmonary Langerhans cell histiocytosis

		# 非朗格汉斯型组织细胞增生
		# 'ORPHA:35687', # Erdheim-Chester disease (Disorder); 多发性骨硬化性组织细胞增生症
		# 'ORPHA:158014', # Rosaï-Dorfman disease (Disorder)

		# 其他组织细胞增生
		# 'ORPHA:168569', # H syndrome (Disorder); OMIM:602782
	],
	'限制性心肌病': [ # 搜索"restrictive cardiomyopathy"
		'ORPHA:75249', # Familial isolated restrictive cardiomyopathy (Disorder)
		'ORPHA:217635', # Familial restrictive cardiomyopathy (Group of disorders)
		'ORPHA:217720', # Non-familial restrictive cardiomyopathy (Group of disorders)
		'ORPHA:217632', # Restrictive cardiomyopathy (Group of disorders)
	],
	'肝豆状核变性': [ # 搜索"Hepatolenticular degeneration"
		'ORPHA:905', # Wilson disease (Disorder)
	],
	'嗜酸性肉芽肿血管炎': [ # 搜索"Churg"
		'ORPHA:183', # Eosinophilic granulomatosis with polyangiitis; Churg-Strauss syndrome (Disorder)
	],
	'肺泡蛋白沉积症': [ # 搜索"pulmonary alveolar proteinosis"
		'ORPHA:747', # Autoimmune pulmonary alveolar proteinosis (Disorder);  OMIM:610910
		'ORPHA:264675', # Hereditary pulmonary alveolar proteinosis (Disorder); OMIM:300770 | OMIM:614370
		'ORPHA:420259', # Secondary pulmonary alveolar proteinosis (Disorder)
		'ORPHA:440427', # Severe early-onset pulmonary alveolar proteinosis due to MARS deficiency (Disorder); OMIM:615486
		'ORPHA:217563', # Neonatal acute respiratory distress due to SP-B deficiency (Disorder); OMIM:265120
		'ORPHA:217566', # Chronic respiratory distress with surfactant metabolism deficiency (Disorder); OMIM:610913
		'ORPHA:440392', # Interstitial lung disease due to SP-C deficiency (Disorder); OMIM:610913
		'ORPHA:440402', # Interstitial lung disease due to ABCA3 deficiency (Disorder); OMIM:610921
	],
	'致心律失常性右室心肌病': [ # 搜索"arrhythmogenic right ventricular dysplasia"
		'ORPHA:247', # Arrhythmogenic right ventricular cardiomyopathy (Group of disorders)
		'ORPHA:217656', #  Familial isolated arrhythmogenic right ventricular dysplasia (Disorder)
		'ORPHA:293910', # Familial isolated arrhythmogenic ventricular dysplasia, right dominant form (Subtype of disorder)
		'ORPHA:98909', # Desminopathy (Disorder); OMIM:601419
	],
	'McCune-Albright综合征': [ # 搜索"McCune"
		'ORPHA:562', # McCune-Albright syndrome (Disorder)
	],
	'心脏离子通道病（长QT间期综合征）': [ # 搜索"long QT"
		'ORPHA:37553', # Andersen-Tawil syndrome; Long QT syndrome type 7 (Disorder); OMIM:170390
		'ORPHA:768', # Familial long QT syndrome (Group of disorders)
		'ORPHA:101016', # Romano-Ward syndrome; Romano-Ward long QT syndrome (Disorder)
		'ORPHA:90647', # Jervell and Lange-Nielsen syndrome; Long QT interval-deafness syndrome (Disorder)
		'ORPHA:65283', # Timothy syndrome; Long QT syndrome-syndactyly syndrome (Disorder); OMIM:601005 | OMIM:618447
	],
	'Ehlers-Danlos综合征': [ # 搜索"Ehlers Danlos"
		'ORPHA:1899', # Arthrochalasia Ehlers-Danlos syndrome (Disorder); OMIM:130060 | OMIM:617821
		'ORPHA:536467', # B3GALT6-related spondylodysplastic Ehlers-Danlos syndrome (Subtype of disorder)
		'ORPHA:75496', # B4GALT7-related spondylodysplastic Ehlers-Danlos syndrome (Subtype of disorder)
		'ORPHA:90354', # Brittle cornea syndrome; Ehlers-Danlos syndrome type VIB (Disorder)
		'ORPHA:230851', # Cardiac-valvular Ehlers-Danlos syndrome (Disorder)
		'ORPHA:287', # Classical Ehlers-Danlos syndrome (Disorder)
		'ORPHA:230839', # Classical-like Ehlers-Danlos syndrome type 1 (Disorder)
		'ORPHA:536532', # Classical-like Ehlers-Danlos syndrome type 2 (Disorder)
		'ORPHA:1901', # Dermatosparaxis Ehlers-Danlos syndrome (Disorder)
		'ORPHA:98249', # Ehlers-Danlos syndrome (Group of disorders))
		'ORPHA:82004', # Ehlers-Danlos syndrome with periventricular heterotopia (Disorder)
		'ORPHA:230857', # Ehlers-Danlos/osteogenesis imperfecta syndrome (Disorder)
		'ORPHA:2295', # Familial articular hypermobility syndrome (Disorder)
		'ORPHA:285', # Hypermobile Ehlers-Danlos syndrome (Disorder)
		'ORPHA:536545', # Kyphoscoliotic Ehlers-Danlos syndrome (Disorder)
		'ORPHA:300179', # Kyphoscoliotic Ehlers-Danlos syndrome due to FKBP22 deficiency (Subtype of disorder)
		'ORPHA:1900', # Kyphoscoliotic Ehlers-Danlos syndrome due to lysyl hydroxylase 1 deficiency (Subtype of disorder)
		'ORPHA:2953', # Musculocontractural Ehlers-Danlos syndrome (Disorder)
		'ORPHA:536516', # Myopathic Ehlers-Danlos syndrome (Disorder)
		'ORPHA:198', # Occipital horn syndrome; Ehlers-Danlos syndrome type IX (Disorder)
		'ORPHA:75392', # Periodontal Ehlers-Danlos syndrome (Disorder)
		'ORPHA:157965', # SLC39A13-related spondylodysplastic Ehlers-Danlos syndrome (Subtype of disorder)
		'ORPHA:536471', # Spondylodysplastic Ehlers-Danlos syndrome (Disorder)
		'ORPHA:286', # Vascular Ehlers-Danlos syndrome (Disorder)
		'ORPHA:230845', # Vascular-like classical Ehlers-Danlos syndrome (Disorder)
		'ORPHA:75497', # X-linked Ehlers-Danlos syndrome (Disorder)
	],
	'结节性硬化症': [ # 搜索"Tuberous sclerosis"
		'ORPHA:805', # Tuberous sclerosis complex (Disorder); OMIM:191100 | OMIM:613254
	],
	'线粒体脑肌病': [ # 搜索"Mitochondrial encephalomyopathy"（线粒体脑肌病）；取名录中列举的常见线粒体脑肌病综合征
		'ORPHA:238329', # Severe X-linked mitochondrial encephalomyopathy (Disorder); OMIM:300816

		# Kearn–Sayre 综合征（KSS）；搜索"KEARNS-SAYRE"
		'ORPHA:480', # Kearns-Sayre syndrome (Disorder); OMIM:530000

		# 慢性进行性眼外肌麻痹（CPEO）；搜索"Mitochondrial progressive external ophthalmoplegia"
		'ORPHA:663', # Mitochondrial DNA-related progressive external ophthalmoplegia (Disorder); 线粒体DNA相关进行性眼外肌麻痹
		'ORPHA:329336', # Adult-onset chronic progressive external ophthalmoplegia with mitochondrial myopathy (Disorder); 成人慢性进行性眼肌麻痹伴线粒体肌病
		'ORPHA:254892', # Autosomal dominant progressive external ophthalmoplegia (后加)
		'ORPHA:254886', # Autosomal recessive progressive external ophthalmoplegia (后加)
		'ORPHA:329314', # Adult-onset multiple mitochondrial DNA deletion syndrome due to DGUOK deficiency (Disorder); OMIM:617070 (后加)

		# 线粒体神经胃肠脑肌病（MINGIE）；搜索"Mitochondrial neurogastrointestinal encephalomyopathy"
		'ORPHA:298', # Mitochondrial neurogastrointestinal encephalomyopathy; MNGIE (Disorder)

		# 线粒体脑肌病伴卒中样发作和乳酸酸中毒（MELAS）; 搜索"MELAS"
		'ORPHA:550', # MELAS; Mitochondrial encephalomyopathy, lactic acidosis and stroke-like episodes (Disorder); OMIM:540000

		# 肌阵挛伴破碎红纤维（MERRF）; 搜索"MERRF"
		'ORPHA:551', # MERRF (Disorder); OMIM:545000

		# 与线粒体DNA相关的Leigh综合征; 搜索"Leigh MITOCHONDRIAL"
		'ORPHA:255210', # Mitochondrial DNA-associated Leigh syndrome (Disorder); OMIM:256000

		# Pearson综合征; 搜索"PEARSON SYNDROME"
		'ORPHA:699', # Pearson syndrome (Disorder); OMIM:557000

		# 神经原性肌无力，共济失调，视网膜色素变性（NARP）; 搜索"NARP"
		'ORPHA:644', # NARP syndrome (Disorder); OMIM:551500

		# Alpers-Huttenlocher综合征; 搜索"Alpers-Huttenlocher"
		'ORPHA:726', # Alpers-Huttenlocher syndrome (Disorder); OMIM:203700
	],
	'线粒体DNA缺失综合征': [ # 搜索"Mitochondrial DNA depletion syndrome"
		'ORPHA:35698', # Mitochondrial DNA depletion syndrome (Group of disorders)
		'ORPHA:254803', # Mitochondrial DNA depletion syndrome, encephalomyopathic form (Group of disorders)
		'ORPHA:1933', # Mitochondrial DNA depletion syndrome, encephalomyopathic form with methylmalonic aciduria (Disorder)
		'ORPHA:255235', # Mitochondrial DNA depletion syndrome, encephalomyopathic form with renal tubulopathy (Disorder)
		'ORPHA:369897', # Mitochondrial DNA depletion syndrome, encephalomyopathic form with variable craniofacial anomalies (Disorder)
		'ORPHA:254871', # Mitochondrial DNA depletion syndrome, hepatocerebral form (Group of disorders)
		'ORPHA:279934', # Mitochondrial DNA depletion syndrome, hepatocerebral form due to DGUOK deficiency (Disorder)
		'ORPHA:363534', # Mitochondrial DNA depletion syndrome, hepatocerebrorenal form (Disorder)
		'ORPHA:254875', # Mitochondrial DNA depletion syndrome, myopathic form (Disorder)
	],
	'Fanconi综合征': [ #
		'ORPHA:3337', # Primary Fanconi syndrome (Disorder)
	],
	'CREST综合征': [
		'ORPHA:90290', # CREST syndrome; OMIM:181750
	],
	'系统性硬化症': [ # 搜索"systemic sclerosis"
		'ORPHA:90291', # Systemic sclerosis (Disorder); OMIM:181750
		'ORPHA:90290', # CREST syndrome; OMIM:181750
		'ORPHA:220407', # Limited systemic sclerosis (Subtype of disorder)
		'ORPHA:220402', # Limited cutaneous systemic sclerosis (Subtype of disorder)
		'ORPHA:220393', # Diffuse cutaneous systemic sclerosis (Subtype of disorder)
		'ORPHA:801', # Scleroderma (Group of disorders)
	],
	'着色性干皮病':[  # 搜索"Xeroderma pigmentosum"
		'ORPHA:910', # Xeroderma pigmentosum (Disorder)
	],

	# pumc_pk
	'Prader-Willi综合征': [
		'ORPHA:739', # Prader-Willi syndrome (Disorder); OMIM:176270
		'ORPHA:177910', # Prader-Willi syndrome due to imprinting mutation (Subtype of disorder)
		'ORPHA:98754', # Prader-Willi syndrome due to maternal uniparental disomy of chromosome 15  (Subtype of disorder)
		'ORPHA:98793', # Prader-Willi syndrome due to paternal 15q11q13 deletion (Subtype of disorder)
		'ORPHA:177901', # Prader-Willi syndrome due to paternal deletion of 15q11q13 type 1 (Subtype of disorder)
		'ORPHA:177904', # Prader-Willi syndrome due to paternal deletion of 15q11q13 type 2 (Subtype of disorder)
		'ORPHA:177907', # Prader-Willi syndrome due to translocation (Subtype of disorder)
	],
	'马方综合征': [
		'ORPHA:558', # Marfan syndrome (Disorder);
		'ORPHA:284963', # Marfan syndrome type 1 (Subtype of disorder)
		'ORPHA:284973', # Marfan syndrome type 2 (Subtype of disorder)
		'ORPHA:284979', # Neonatal Marfan syndrome (Disorder)
	],
	'肌萎缩侧索硬化':[
		'ORPHA:803', # Amyotrophic lateral sclerosis (Disorder)
		'ORPHA:357043', # Amyotrophic lateral sclerosis type 4
		'ORPHA:300605', # Juvenile amyotrophic lateral sclerosis

		# 'ORPHA:52430', # Inclusion body myopathy with Paget disease of bone and frontotemporal dementia; Pagetoid amyotrophic lateral sclerosis
		# 'ORPHA:90020', # Amyotrophic lateral sclerosis-parkinsonism-dementia complex; OMIM:105500
		# 'ORPHA:275872', # Frontotemporal dementia with motor neuron disease; Frontotemporal dementia with amyotrophic lateral sclerosis; FDT-ALS
	],
	'多系统萎缩':[
		'ORPHA:102', # Multiple system atrophy (Disorder)
		'ORPHA:227510', # Multiple system atrophy, cerebellar type (Subtype of disorder)
		'ORPHA:98933', # Multiple system atrophy, parkinsonian type (Subtype of disorder)
	],
	'Alport综合征':[
		'ORPHA:63', # Alport syndrome (Disorder)
		'ORPHA:88918', # Autosomal dominant Alport syndrome (Subtype of disorder)
		'ORPHA:88919', # Autosomal recessive Alport syndrome
		'ORPHA:88917', # X-linked Alport syndrome

		# 'ORPHA:86818', # Alport syndrome-intellectual disability-midface hypoplasia-elliptocytosis syndrome (Disorder); OMIM:300194
		# 'ORPHA:1019', # Epstein syndrome; Alport syndrome with macrothrombocytopenia
		# 'ORPHA:1984', # Fechtner syndrome (Subtype of disorder); Alport syndrome with leukocyte inclusions and macrothrombocytopenia
		# 'ORPHA:1018', # X-linked Alport syndrome-diffuse leiomyomatosis (Subtype of disorder)
	],
	'阵发性睡眠性血红蛋白尿':[
		'ORPHA:447', # Paroxysmal nocturnal hemoglobinuria (Disorder)
	],
	'尼曼匹克病':[
		'ORPHA:77292', # Niemann-Pick disease type A (Disorder)
		'ORPHA:77293', # Niemann-Pick disease type B (Disorder)
		'ORPHA:646', # Niemann-Pick disease type C (Disorder)
		'ORPHA:216986', # Niemann-Pick disease type C, adult neurologic onset (Subtype of disorder)
		'ORPHA:216981', # Niemann-Pick disease type C, juvenile neurologic onset (Subtype of disorder)
		'ORPHA:216978', # Niemann-Pick disease type C, late infantile neurologic onset (Subtype of disorder)
		'ORPHA:216975 ', # Niemann-Pick disease type C, severe early infantile neurologic onset (Subtype of disorder)
		'ORPHA:216972', # Niemann-Pick disease type C, severe perinatal form (Subtype of disorder)
		'ORPHA:79289', # Niemann-Pick disease type D (Disorder)
	],

	### todo 2020 11 27 增加协和MDT数据的诊断测试 ###
	'阴道闭锁': [
		'ORPHA:3411',
		'ORPHA:65681',
	],
	'家族性高胆固醇血症': [],
	# CRESR综合征 上面已经有
	'纯合子家族性高胆固醇血症': [
		'ORPHA:391665',
	],
	'PROTEASOME-ASSOCIATED AUTOINFLAMMATORY SYNDROME 2':[],
	'Activated PI3K-delta syndrome':[
		'ORPHA:397596',
		'ORPHA:101972',  # 联合免疫缺陷疾病中的活化PI3K-δ综合征 为了增加可实验性，因为这一个只有orpha的 不能做测试
	],
	'Primary angiitis of the central nervous system':[
		'ORPHA:140989',
	],
	'粘多糖病第Ⅱ型': [
		'ORPHA:580',
	],
	'戈谢病':[
		'ORPHA:355',
		'ORPHA:77260',
		'ORPHA:77259',
		'ORPHA:85212',
		'ORPHA:2072',
		'ORPHA:77261',
		'ORPHA:309252',
	],
	'脊髓小脑性共济失调': [
		'ORPHA:98761',  # 脊髓小脑性共济失调第10型
		'ORPHA:98762',  # 脊髓小脑性共济失调第12型
		'ORPHA:98763',  # 脊髓小脑性共济失调第14型
		'ORPHA:98764',  # 脊髓小脑性共济失调第27型
		'ORPHA:98767',  # 脊髓小脑性共济失调第11型
		'ORPHA:98768',  # 脊髓小脑性共济失调第13型
		'ORPHA:98773',  # 脊髓小脑性共济失调第21型
		'ORPHA:217012',  # 脊髓小脑共济失调31
		'ORPHA:98760',  # 脊髓小脑性共济失调第8型
		'ORPHA:98766',  # 脊髓小脑性共济失调第5型
		'ORPHA:98772',  # 脊髓小脑性共济失调第19/22型
		'ORPHA:101112',  # 脊髓小脑共济失调26（SCA26）
		'ORPHA:208513',  # 脊髓小脑共济失调29；SCA29
		'ORPHA:276198',  # 脊髓小脑共济失调36；SCA36
		'ORPHA:98769',  # 脊髓小脑性共济失调第15/16型
		'ORPHA:101109',  # 脊髓小脑性共济失调28（SCA28）
		'ORPHA:276193',  # 脊髓小脑性共济失调35; SCA35
		'ORPHA:423275',  # 脊髓小脑性共济失调40; SCA40
		'ORPHA:423296',  # 脊髓小脑性共济失调38（SCA38）
		'ORPHA:458798',  # 脊髓小脑性共济失调41（SCA41）
		# 'OMIM:607136',  #脊髓小脑运动失调17型
		'ORPHA:101108',  # 脊髓小脑的共济失调23; SCA23
		# 'OMIM:164400',  #脊髓小脑运动失调1型
		# 'OMIM:183086',  #脊髓小脑运动失调6型
		# 'OMIM:183090',  #脊髓小脑运动失调2型
		'ORPHA:98765',  # 脊髓小脑性共济失调第4型
		'ORPHA:98771',  # 脊髓小脑性共济失调第18型
		# 'OMIM:133190',  #红斑角皮病伴共济失调(Spinocerebellar ataxia type 34) ?? 暂时不知道这个要不要加上去
		# 'OMIM:164500',  #常染色体显性脊髓小脑运动失调7型
		# 'OMIM:109150',  #脊髓小脑运动失调3型
		# 'OMIM:183090',  #
	],
	'Nodular fasciitis': [
		'ORPHA:477742',  #！*！
		'ORPHA:592',     #！*！
		'ORPHA:3165',    #！*！
	],
	'高安动脉炎,无脉病,大动脉炎':[
		'ORPHA:3287',
	],
	'Aicardi Goutieres综合征': [],
	'缺铁性红细胞贫血伴b细胞免疫缺乏，周期性发烧和发育迟缓':[
		'ORPHA:369861',
	],
	#Castleman病 已经有了

	'先天性脂肪瘤样增生，血管畸形和表皮痣综合征': [
		'ORPHA:140944',
	],
	'先天性多发关节挛缩综合征': [
		'ORPHA:1143',
		'ORPHA:319332',
	],
	'macrodactyly':[
		'ORPHA:295044',
		'ORPHA:295047',
		'ORPHA:295239',
		'ORPHA:295241',
		'ORPHA:295243',
		'ORPHA:295245',
	],
	#McCune-Albright综合征 已经有了
	'肾上腺皮质癌':[
		'ORPHA:1501',
	],
	'Oncogenic osteomalacia':[
		'ORPHA:352540',
	],
	# 肌萎缩侧索硬化 上面已经有了
	# Fabry病 上面已经有了
	'红细胞生成性原卟啉病':[
		'ORPHA:79278',
		'ORPHA:738',
	],
	'瓦登伯格综合征':[
		'ORPHA:896',
		'ORPHA:3440',
		'ORPHA:894',
		'ORPHA:897',
		'ORPHA:895',
	],
	# SAPHO 上面已经有了
	# APDS，Activated PI3K-delta syndrome 上面已经有了
	# 结节性硬化症 上面已经有了
	'Primary hypertrophic osteoarthropathy': [
		'ORPHA:248095',
	],
	'脊髓性肌萎缩症': [
		'ORPHA:70',     #脊髓性肌萎缩症
		'ORPHA:83330',  #婴儿脊髓性肌萎缩
		'ORPHA:83419',  #青少年脊髓型肌萎缩
		'ORPHA:209341', #脊髓性肌萎缩（下肢受累，常染色体显性遗传）-1；SMALED1
		'ORPHA:363454', #脊髓性肌萎缩（下肢受累，常染色体显性遗传）-2；SMALED2
	],
	# 蕈样肉芽肿 上面已经有了

	# 2020 11 30 为了做 CCRD所有疾病的测试 #
	#'21-羟化酶缺乏症': ['21-羟化酶缺乏症'],
	'白化病': [
		'ORPHA:54',
		'ORPHA:55',
		'ORPHA:79433',
		'ORPHA:79434',
		'ORPHA:79435',
		'ORPHA:79432',
		'ORPHA:167',
		'ORPHA:998',
		'ORPHA:381',
		'ORPHA:79431',
		'ORPHA:352745',
		'ORPHA:98835',
	],
	#'Alport 综合征': [''],
	#'肌萎缩侧索硬化': [''],
	'天使综合征': [
		'ORPHA:72',
		'ORPHA:98794',
		'ORPHA:98795',
	],
	'精氨酸酶缺乏症': [
		'ORPHA:90',
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
	#'特发性肺动脉高压': [''],
	'特发性肺纤维化': [],
	#'IgG4 相关性疾病': [''],
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
		d = CLASS_TO_ORPHANET
		for diag_str, diag_codes in d.items():
			assert no_strip(diag_str)
			for diag_code in diag_codes:
				assert no_strip(diag_code)
			assert no_repeat(diag_codes)
	check()

