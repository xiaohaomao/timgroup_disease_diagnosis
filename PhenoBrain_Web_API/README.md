# 说明
- 接口请求域名：http://www.phenobrain.cs.tsinghua.edu.cn
- Phenobrain网站，仅供参考：[http://www.phenobrain.cs.tsinghua.edu.cn/pc](http://www.phenobrain.cs.tsinghua.edu.cn/pc)

# API概览
- [hpo-tree-init](#hpo-tree-init)：初始化表型树
- [hpo-child](#hpo-child)：获取单个表型的子节点列表
- [hpo-child-many](#hpo-child-many)：获取多个表型的子节点
- [hpo-detail](#hpo-detail)：获取表型的详细信息
- [disease-detail](#disease-detail)：获取疾病的详细信息
- [disease-list-detail](#disease-list-detail)：获取疾病列表中每个疾病的详细信息
- [predict](#predict)：输入表型集合和模型选择，输出罕见病列表；后台返回任务ID
- [query-predict-result](#query-predict-result)：根据任务ID，获取罕见病预测结果
- [extract-hpo](#extract-hpo)：输入自由文本，输出抽取的表型；后台返回任务ID
- [query-extract-hpo-result](#query-extract-hpo-result)：根据任务ID，获取表型的抽取结果
- [search-hpo](#search-hpo)：表型搜索
- [search-dis](#search-dis)：疾病搜索
- [数据结构](#DataStructure)

## API修改说明
原来为了nginx规则匹配的方便，随便给API取了暂时的名字，现将以下API更名：

| 原API | 新API |
|------|----|----|
| predict2 | extract-hpo |
| query-predict-result2 | query-extract-hpo-result |
| search-hpo2 | search-dis |

请在代码中搜索相应的ajax调用，将API更名。

<h1 id="hpo-tree-init"> hpo-tree-init </h1>

## 接口描述
- 获取初始的表型树信息
- 请求方法：GET

## 输入参数
- 无

## 输出参数
- Array of [HpoTreeNode](#HpoTreeNode)

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/hpo-tree-init
```

输出示例：

```
[
  {
    "hpo": "HP:0000001",
    "CNS_NAME": "",
    "ENG_NAME": "All",
    "children": [
      {
        "hpo": "HP:0000005",
        "CNS_NAME": "遗传模式",
        "ENG_NAME": "Mode of inheritance",
        "children": []
      },
      {
        "hpo": "HP:0000118",
        "CNS_NAME": "表型异常",
        "ENG_NAME": "Phenotypic abnormality",
        "children": []
      },
      {
        "hpo": "HP:0012823",
        "CNS_NAME": "临床调节因素",
        "ENG_NAME": "Clinical modifier",
        "children": []
      },
      {
        "hpo": "HP:0031797",
        "CNS_NAME": "",
        "ENG_NAME": "Clinical course",
        "children": []
      },
      {
        "hpo": "HP:0032223",
        "CNS_NAME": "",
        "ENG_NAME": "Blood group",
        "children": []
      },
      {
        "hpo": "HP:0032443",
        "CNS_NAME": "",
        "ENG_NAME": "Past medical history",
        "children": []
      },
      {
        "hpo": "HP:0040279",
        "CNS_NAME": "",
        "ENG_NAME": "Frequency",
        "children": []
      }
    ]
  }
]
```

<h1 id="hpo-child"> hpo-child </h1>

## 接口描述
- 获取单个表型的子节点列表
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| hpo | String | 父节点的HPO编码 |

## 输出参数
- Array of [HpoChildNode](#HpoChildNode)

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/hpo-child?hpo=HP:0000478
```

输出示例：

```
[
  {
    "CHILD": [
      "HP:0000553",
      "HP:0000589",
      "HP:0000591",
      "HP:0000667",
      "HP:0004328",
      "HP:0004329",
      "HP:0008047",
      "HP:0008056",
      "HP:0010727",
      "HP:0100012",
      "HP:0100886",
      "HP:0100887"
    ],
    "CNS_NAME": "眼部形态异常",
    "CODE": "HP:0012372",
    "ENG_NAME": "Abnormal eye morphology"
  },
  {
    "CHILD": [
      "HP:0000496",
      "HP:0000501",
      "HP:0000504",
      "HP:0000508",
      "HP:0000539",
      "HP:0000632",
      "HP:0007686",
      "HP:0011885",
      "HP:0012632",
      "HP:0025401",
      "HP:0025590",
      "HP:0030453",
      "HP:0030637",
      "HP:0030800",
      "HP:0031590",
      "HP:0100533",
      "HP:0200026"
    ],
    "CNS_NAME": "眼部生理异常",
    "CODE": "HP:0012373",
    "ENG_NAME": "Abnormal eye physiology"
  }
]
```

<h1 id="hpo-child-many"> hpo-child-many </h1>

## 接口描述
- 获取多个表型的子节点
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| hpoList | Array of String | 需要查询子节点的表型列表 |

## 输出参数
- {HPO_CODE: Array of [SearchHpoInfo](#SearchHpoInfo)} 
- HPO_CODE 在输入的表型列表的子节点的并集中

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/hpo-child-many?
hpoList[]=HP:0000256
&hpoList[]=HP:0002679
```

输出示例

```
{
  "HP:0002681": [
    {
      "CNS_NAME": "J形蝶鞍",
      "CODE": "HP:0002680",
      "ENG_NAME": "J-shaped sella turcica"
    },
    {
      "CNS_NAME": "桥接蝶鞍",
      "CODE": "HP:0005449",
      "ENG_NAME": "Bridged sella turcica"
    },
    {
      "CNS_NAME": "长蝶鞍",
      "CODE": "HP:0005463",
      "ENG_NAME": "Elongated sella turcica"
    },
    {
      "CNS_NAME": "鞋形蝶鞍",
      "CODE": "HP:0005723",
      "ENG_NAME": "Shoe-shaped sella turcica"
    },
    {
      "CNS_NAME": "蝶鞍扁平",
      "CODE": "HP:0100857",
      "ENG_NAME": "Flat sella turcica"
    }
  ],
  "HP:0002690": [],
  "HP:0004481": [],
  "HP:0004482": [],
  "HP:0004488": [],
  "HP:0005490": [],
  "HP:0010538": [],
  "HP:0040304": []
}
```

<h1 id="hpo-detail"> hpo-detail </h1>

## 接口描述
- 获取表型的详细信息
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| hpo | String | 表型的HPO编码 |
| projection | Array of String | 从[CODE, CNS\_NAME, ENG\_NAME, CNS\_DEF, ENG\_DEF, SYNONYM, REL\_DIS, ALL\_PATH, TOTAL\_DIS\_NUM]中任选，输出将包含所选的字段的信息 |

## 输出参数
- [HpoInfo](#HpoInfo)

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/hpo-detail?
hpo=HP:0000256
&projection[]=CODE
&projection[]=CNS_NAME
&projection[]=ENG_NAME
&projection[]=CNS_DEF
&projection[]=ENG_DEF
&projection[]=SYNONYM
&projection[]=REL_DIS
&projection[]=ALL_PATH
&projection[]=TOTAL_DIS_NUM
```

输出示例

```
{
  "CNS_DEF": "枕额（头）周长大于年龄和性别相匹配的正常值的第97百分位数。或者头盖骨尺寸显得增大。",
  "CNS_NAME": "大头畸形",
  "CODE": "HP:0000256",
  "ENG_DEF": "\"Occipitofrontal (head) circumference greater than 97th centile compared to appropriate, age matched, sex-matched normal standards. Alternatively, a apparently increased size of the cranium.\" [PMID:19125436]",
  "ENG_NAME": "Macrocephaly",
  "PATH_DICT": {
    "HP:0000001": [
      "HP:0000118"
    ],
    "HP:0000118": [
      "HP:0000152",
      "HP:0000924"
    ],
    "HP:0000152": [
      "HP:0000234"
    ],
    "HP:0000234": [
      "HP:0000929"
    ],
    "HP:0000240": [
      "HP:0040194"
    ],
    "HP:0000256": [],
    "HP:0000924": [
      "HP:0011842"
    ],
    "HP:0000929": [
      "HP:0000240"
    ],
    "HP:0009121": [
      "HP:0000929"
    ],
    "HP:0011842": [
      "HP:0009121"
    ],
    "HP:0040194": [
      "HP:0000256"
    ]
  },
  "PATH_INFO_DICT": {
    "HP:0000001": {
      "ENG_NAME": "All"
    },
    "HP:0000118": {
      "CNS_NAME": "表型异常",
      "ENG_NAME": "Phenotypic abnormality"
    },
    "HP:0000152": {
      "CNS_NAME": "头部和颈部的异常",
      "ENG_NAME": "Abnormality of head or neck"
    },
    "HP:0000234": {
      "CNS_NAME": "头部异常",
      "ENG_NAME": "Abnormality of the head"
    },
    "HP:0000240": {
      "CNS_NAME": "头骨的大小异常",
      "ENG_NAME": "Abnormality of skull size"
    },
    "HP:0000256": {
      "CNS_NAME": "大头畸形",
      "ENG_NAME": "Macrocephaly"
    },
    "HP:0000924": {
      "CNS_NAME": "骨骼系统异常",
      "ENG_NAME": "Abnormality of the skeletal system"
    },
    "HP:0000929": {
      "CNS_NAME": "颅骨异常",
      "ENG_NAME": "Abnormal skull morphology"
    },
    "HP:0009121": {
      "CNS_NAME": "中轴骨架形态异常",
      "ENG_NAME": "Abnormal axial skeleton morphology"
    },
    "HP:0011842": {
      "CNS_NAME": "骨骼形态异常",
      "ENG_NAME": "Abnormality of skeletal morphology"
    },
    "HP:0040194": {
      "CNS_NAME": "头围增加",
      "ENG_NAME": "Increased head circumference"
    }
  },
  "REL_DIS": 408,
  "SYNONYM": [
    "Big calvaria",
    "Big cranium",
    "Big head",
    "Big skull",
    "Increased size of cranium",
    "Increased size of head",
    "Increased size of skull",
    "Large calvaria",
    "Large cranium",
    "Large head",
    "Large head circumference",
    "Large skull",
    "Macrocrania",
    "Megacephaly"
  ],
  "TOTAL_DIS_NUM": 9260
}
```

<h1 id="disease-detail"> disease-detail </h1>

## 接口描述
- 获取疾病的详细信息
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| disCode | String | 疾病的OMIM、ORPHANET、CCRD或RD编码 |
| paHpoList | Array of String | 所选择的病人表型，用于计算“表型注释”模块中，病人表型与疾病表型的关系 |

## 输出参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| CODE | String | 疾病编码 |
| CNS\_NAME | String | 疾病的中文名称 |
| ENG\_NAME | String | 疾病的英文名称 |
| GENE | String | 疾病对应的基因信息 |
| ANNO\_HPO\_PROB | Array | 疾病的表型概率信息 |
| ANNO\_HPO\_DIFF | Object | 疾病表型集合与病人表型集合的精确匹配、泛化匹配、细化匹配等信息 |
| ANNO\_HPO | Array of String | 疾病的表型列表 |
| HPO_INFO | Object | 涉及的表型的信息 |
| SOURCE\_CODES | Array of String | 疾病的其他等价编码 |

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/disease-detail?
disCode=RD:3502
&paHpoList[]=HP:0001913
&paHpoList[]=HP:0008513
&paHpoList[]=HP:0001123
&paHpoList[]=HP:0000365
&paHpoList[]=HP:0002857
&paHpoList[]=HP:0001744
```

输出示例

```
{
  "ANNO_HPO": [
    "HP:0008341",
    "HP:0011002",
    "HP:0001873",
    "HP:0002757",
    "HP:0000689",
    "HP:0000303",
    "HP:0000648",
    "HP:0002857",
    "HP:0010885",
    "HP:0001508",
    "HP:0005930",
    "HP:0009830",
    "HP:0003034",
    "HP:0002240",
    "HP:0003148",
    "HP:0001433",
    "HP:0002653",
    "HP:0006482",
    "HP:0001978",
    "HP:0008153",
    "HP:0000572",
    "HP:0007807",
    "HP:0002135",
    "HP:0000091",
    "HP:0004322",
    "HP:0002514",
    "HP:0004349",
    "HP:0004437",
    "HP:0001263",
    "HP:0001744",
    "HP:0000007",
    "HP:0000670",
    "HP:0000505",
    "HP:0001903",
    "HP:0001249"
  ],
  "ANNO_HPO_DIFF": {
    "EXACT": [
      "HP:0002857",
      "HP:0001744"
    ],
    "GENERAL": [
      [
        "HP:0000505",
        [
          [
            "HP:0001123",
            1
          ]
        ]
      ]
    ],
    "OTHER": [
      "HP:0008341",
      "HP:0011002",
      "HP:0001873",
      "HP:0002757",
      "HP:0000689",
      "HP:0000303",
      "HP:0000648",
      "HP:0010885",
      "HP:0001508",
      "HP:0005930",
      "HP:0009830",
      "HP:0003034",
      "HP:0002240",
      "HP:0003148",
      "HP:0001433",
      "HP:0002653",
      "HP:0006482",
      "HP:0001978",
      "HP:0008153",
      "HP:0000572",
      "HP:0007807",
      "HP:0002135",
      "HP:0000091",
      "HP:0004322",
      "HP:0002514",
      "HP:0004349",
      "HP:0004437",
      "HP:0001263",
      "HP:0000007",
      "HP:0000670",
      "HP:0001903",
      "HP:0001249"
    ],
    "SPECIFIC": []
  },
  "ANNO_HPO_PROB": [
    [
      "HP:0000007",
      null
    ],
    [
      "HP:0000572",
      null
    ],
    [
      "HP:0000689",
      "30%-79%"
    ],
    [
      "HP:0001249",
      "80%-99%"
    ],
    [
      "HP:0001433",
      null
    ],
    [
      "HP:0001903",
      "80%-99%"
    ],
    [
      "HP:0001978",
      null
    ],
    [
      "HP:0002135",
      null
    ],
    [
      "HP:0003034",
      null
    ],
    [
      "HP:0003148",
      null
    ],
    [
      "HP:0004322",
      null
    ],
    [
      "HP:0004437",
      null
    ],
    [
      "HP:0007807",
      null
    ],
    [
      "HP:0008153",
      null
    ],
    [
      "HP:0008341",
      null
    ],
    [
      "HP:0011002",
      "80%-99%"
    ],
    [
      "HP:0000091",
      "80%-99%"
    ],
    [
      "HP:0000303",
      "30%-79%"
    ],
    [
      "HP:0000505",
      "5%-29%"
    ],
    [
      "HP:0000648",
      "5%-29%"
    ],
    [
      "HP:0000670",
      "30%-79%"
    ],
    [
      "HP:0001263",
      "80%-99%"
    ],
    [
      "HP:0001508",
      "80%-99%"
    ],
    [
      "HP:0001744",
      "80%-99%"
    ],
    [
      "HP:0001873",
      "30%-79%"
    ],
    [
      "HP:0002240",
      "80%-99%"
    ],
    [
      "HP:0002514",
      "30%-79%"
    ],
    [
      "HP:0002653",
      "80%-99%"
    ],
    [
      "HP:0002757",
      "80%-99%"
    ],
    [
      "HP:0002857",
      "80%-99%"
    ],
    [
      "HP:0004349",
      "80%-99%"
    ],
    [
      "HP:0005930",
      "80%-99%"
    ],
    [
      "HP:0006482",
      "30%-79%"
    ],
    [
      "HP:0009830",
      "30%-79%"
    ],
    [
      "HP:0010885",
      "80%-99%"
    ]
  ],
  "CNS_NAME": "肾小管性酸中毒III型",
  "CODE": "RD:3502",
  "ENG_NAME": "Osteopetrosis with renal tubular acidosis",
  "GENE": [
    "CA2"
  ],
  "HPO_INFO": {
    "HP:0000007": {
      "CNS_NAME": "常染色体隐形遗传",
      "ENG_NAME": "Autosomal recessive inheritance"
    },
    "HP:0000091": {
      "CNS_NAME": "肾小管异常",
      "ENG_NAME": "Abnormal renal tubule morphology"
    },
    "HP:0000303": {
      "CNS_NAME": "下颌前突畸形",
      "ENG_NAME": "Mandibular prognathia"
    },
    "HP:0000365": {
      "CNS_NAME": "听力障碍",
      "ENG_NAME": "Hearing impairment"
    },
    "HP:0000505": {
      "CNS_NAME": "视觉障碍",
      "ENG_NAME": "Visual impairment"
    },
    "HP:0000572": {
      "CNS_NAME": "视力下降",
      "ENG_NAME": "Visual loss"
    },
    "HP:0000648": {
      "CNS_NAME": "视神经萎缩",
      "ENG_NAME": "Optic atrophy"
    },
    "HP:0000670": {
      "CNS_NAME": "龋齿",
      "ENG_NAME": "Carious teeth"
    },
    "HP:0000689": {
      "CNS_NAME": "牙齿错位咬合",
      "ENG_NAME": "Dental malocclusion"
    },
    "HP:0001123": {
      "CNS_NAME": "视野缺损",
      "ENG_NAME": "Visual field defect"
    },
    "HP:0001249": {
      "CNS_NAME": "智力残疾",
      "ENG_NAME": "Intellectual disability"
    },
    "HP:0001263": {
      "CNS_NAME": "全面发育迟缓",
      "ENG_NAME": "Global developmental delay"
    },
    "HP:0001433": {
      "CNS_NAME": "肝脾肿大",
      "ENG_NAME": "Hepatosplenomegaly"
    },
    "HP:0001508": {
      "CNS_NAME": "未能茁壮成长",
      "ENG_NAME": "Failure to thrive"
    },
    "HP:0001744": {
      "CNS_NAME": "脾肿大",
      "ENG_NAME": "Splenomegaly"
    },
    "HP:0001873": {
      "CNS_NAME": "血小板减少",
      "ENG_NAME": "Thrombocytopenia"
    },
    "HP:0001903": {
      "CNS_NAME": "贫血",
      "ENG_NAME": "Anemia"
    },
    "HP:0001913": {
      "CNS_NAME": "粒细胞减少",
      "ENG_NAME": "Granulocytopenia"
    },
    "HP:0001978": {
      "CNS_NAME": "骨髓外造血",
      "ENG_NAME": "Extramedullary hematopoiesis"
    },
    "HP:0002135": {
      "CNS_NAME": "基底节钙化",
      "ENG_NAME": "Basal ganglia calcification"
    },
    "HP:0002240": {
      "CNS_NAME": "肝脏肿大",
      "ENG_NAME": "Hepatomegaly"
    },
    "HP:0002514": {
      "CNS_NAME": "脑钙化",
      "ENG_NAME": "Cerebral calcification"
    },
    "HP:0002653": {
      "CNS_NAME": "骨痛",
      "ENG_NAME": "Bone pain"
    },
    "HP:0002757": {
      "CNS_NAME": "复发性骨折",
      "ENG_NAME": "Recurrent fractures"
    },
    "HP:0002857": {
      "CNS_NAME": "膝外翻",
      "ENG_NAME": "Genu valgum"
    },
    "HP:0003034": {
      "CNS_NAME": "骨干硬化症",
      "ENG_NAME": "Diaphyseal sclerosis"
    },
    "HP:0003148": {
      "CNS_NAME": "血清酸性磷酸酶升高",
      "ENG_NAME": "Elevated serum acid phosphatase"
    },
    "HP:0004322": {
      "CNS_NAME": "身材矮小",
      "ENG_NAME": "Short stature"
    },
    "HP:0004349": {
      "CNS_NAME": "骨密度降低",
      "ENG_NAME": "Reduced bone mineral density"
    },
    "HP:0004437": {
      "CNS_NAME": "颅骨质增生",
      "ENG_NAME": "Cranial hyperostosis"
    },
    "HP:0005930": {
      "CNS_NAME": "骨骺异常",
      "ENG_NAME": "Abnormality of epiphysis morphology"
    },
    "HP:0006482": {
      "CNS_NAME": "牙齿形态异常",
      "ENG_NAME": "Abnormality of dental morphology"
    },
    "HP:0007807": {
      "CNS_NAME": "视神经受压",
      "ENG_NAME": "Optic nerve compression"
    },
    "HP:0008153": {
      "CNS_NAME": "低钾性周期性麻痹",
      "ENG_NAME": "Periodic hypokalemic paresis"
    },
    "HP:0008341": {
      "CNS_NAME": "远端肾小管性酸中毒",
      "ENG_NAME": "Distal renal tubular acidosis"
    },
    "HP:0008513": {
      "CNS_NAME": "双侧传导性听觉障碍",
      "ENG_NAME": "Bilateral conductive hearing impairment"
    },
    "HP:0009830": {
      "CNS_NAME": "周围神经病",
      "ENG_NAME": "Peripheral neuropathy"
    },
    "HP:0010885": {
      "CNS_NAME": "无菌性骨坏死",
      "ENG_NAME": "Avascular necrosis"
    },
    "HP:0011002": {
      "CNS_NAME": "骨硬化",
      "ENG_NAME": "Osteopetrosis"
    }
  },
  "SOURCE_CODES": [
    "OMIM:259730",
    "ORPHA:2785"
  ]
}
```

<h1 id="disease-list-detail"> disease-list-detail </h1>

## 接口描述
- 获取疾病列表中每个疾病的详细信息
- 请求方法：POST

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| diseaseList | Array of String | 疾病列表 |

## 输出参数
- {DIS_CODE: [SearchDisInfo](#SearchDisInfo)}

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/disease-list-detail

Request Payload:
{"diseaseList":["RD:454","RD:8366"]}
```

输出示例

```
{
  "RD:454": {
    "CNS_NAME": "进行性骨干发育不良",
    "ENG_NAME": "Camurati-Engelmann disease",
    "SOURCE_CODES": [
      "OMIM:131300",
      "ORPHA:1328"
    ]
  },
  "RD:8366": {
    "CNS_NAME": "关节-齿-骨发育不良",
    "ENG_NAME": "Acroosteolysis dominant type",
    "SOURCE_CODES": [
      "OMIM:102500",
      "ORPHA:955"
    ]
  }
}
```




<h1 id="predict"> predict </h1>

## 接口描述
- 输入表型集合和模型选择，输出罕见病列表；返回任务ID，用于之后查询预测结果
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| model | String | 从['Ensemble', 'ICTO (A)', 'ICTO (U)', 'PPO', 'CNB', 'MLP (M)', 'MinIC', 'Res', 'BOQA', 'GDDP', 'RBP', 'Lin', 'JC', 'SimUI', 'TO', 'Cosine', 'RDD'] 中任选|
| hpoList | Array of String | 输入的病人表型集合 |
| topk | Integer | 返回可能患有的疾病列表的大小，最大值为200，高于200将被自动设为200 |

## 输出参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| TASK\_ID | String | 预测任务的ID，用于之后查询预测结果 |

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/predict?
model=Ensemble
&hpoList[]=HP:0001913
&hpoList[]=HP:0008513
&hpoList[]=HP:0001123
&hpoList[]=HP:0000365
&hpoList[]=HP:0002857
&hpoList[]=HP:0001744
&topk=5
```

输出示例

```
{"TASK_ID":"a4fecf25-9eb6-4ac6-a02a-0eae329c9615"}
```

<h1 id="query-predict-result"> query-predict-result </h1>

## 接口描述
- 根据任务ID，获取罕见病预测结果
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| taskId | String | 任务ID |

## 输出参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| result | Array | 根据匹配得分由高到低排序的疾病列表 |
| state | String | 当前任务的执行状态：'SUCCESS'表示预测算法执行完毕；'MODEL\_INIT'表示正在初始化模型；'MODEL\_PREDICT'表示预测算法正在运行 |

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/query-predict-result?
taskId=a4fecf25-9eb6-4ac6-a02a-0eae329c9615
```

输出示例

```
{
  "result": [
    {
      "CODE": "RD:8366",
      "SCORE": 0.9989560835133189
    },
    {
      "CODE": "RD:6799",
      "SCORE": 0.9988120950323974
    },
    {
      "CODE": "RD:3502",
      "SCORE": 0.9988120950323974
    },
    {
      "CODE": "RD:7158",
      "SCORE": 0.9986321094312455
    },
    {
      "CODE": "RD:7367",
      "SCORE": 0.9984881209503239
    }
  ],
  "state": "SUCCESS"
}
```

<h1 id="extract-hpo"> extract-hpo </h1>

## 接口描述
- 输入自由文本，输出抽取的表型；后台返回任务ID
- 请求方法：POST
- 注：原来的API为`predict2`

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| text | String | 病人的描述文本 |
| method | String | 从['HPO/CHPO', 'CHPO-UMLS', 'CText2Hpo（S）']中任选 |
| threshold | String | 得分阈值，低于该得分的匹配结果将被过滤掉，当method=='CText2Hpo（S）'时才有用 |

## 输出参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| TASK\_ID | String | 任务ID |

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/extract-hpo

Request Payload:
{
  "text":"现病史：患者于2001左右无明显诱因出现四肢疼痛，以手掌、脚掌为著，天热、感冒发热、运动后加重，休息后可缓解，主要为烧灼痛、胀痛，无明显手脚麻木、感觉减退。并伴有皮肤腰骶、大腿内侧散在零星针尖样大小、紫红色、不高于皮肤表面皮疹，无压痛无瘙痒。近1年前出现两次眩晕，发作时主要为视物旋转，偶伴有恶心，呕吐，无耳鸣等伴随症状，就诊于北京医院诊断为“耳石症”，行复位治疗后好转。近半年患者偶尔有胸闷、憋气、点状针刺样胸前区疼痛症状，发作与体位、活动无明显关联，休息约10-20分钟后可缓解，未曾服用硝酸酯类药物。四肢疼痛较前无明显加重，偶有过电感，无明显麻木。1周前眩晕症状再次出现，外院就诊未予明确诊断。此次为行进一步检查收入我科。 近半年精神、食欲、睡眠可，小便可，近2月大便性状改变，每日2-3次不成形糊状便，颜色不详。体重无明显下降。\n目前诊断：眩晕微循环障碍不除外；听力减退；心肌受累；高血压病（II级 中危组）右侧大脑后动脉纤细；双眼青光眼不除外；右面部色素痣",
  "method":"HPO/CHPO",
  "threshold":""
}
```

输出示例

```
{"TASK_ID":"a6ffd993-7a4a-4164-805b-68e24db80227"}
```

<h1 id="query-extract-hpo-result"> query-extract-hpo-result </h1>

## 接口描述
- 根据任务ID，获取表型的抽取结果
- 请求方法：GET
- 注：原来的API为`query-predict-result2`

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| taskId | String | 任务ID |

## 输出参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| result | Array | 根据匹配得分由高到低排序的疾病列表 |
| state | String | 当前任务的执行状态：'SUCCESS'表示预测算法执行完毕；'PROCESS\_TEXT'表示正在预处理文本；'EXTRACT\_HPO'表示表型抽取算法正在运行 |

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/query-extract-hpo-result?taskId=a6ffd993-7a4a-4164-805b-68e24db80227
```

输出示例

```
{
  "result": {
    "HPO_LIST": [
      "HP:0012531",
      "HP:0001945",
      "HP:0000988",
      "HP:0002321",
      "HP:0002018",
      "HP:0002013",
      "HP:0012531",
      "HP:0002321",
      "HP:0000822",
      "HP:0007481",
      "HP:0003764"
    ],
    "HPO_TO_INFO": {
      "HP:0000822": {
        "CNS_NAME": "高血压",
        "ENG_NAME": "Hypertension",
        "LABEL_CAT_ID": 641
      },
      "HP:0000988": {
        "CNS_NAME": "皮疹",
        "ENG_NAME": "Skin rash",
        "LABEL_CAT_ID": 773
      },
      "HP:0001945": {
        "CNS_NAME": "发热",
        "ENG_NAME": "Fever",
        "LABEL_CAT_ID": 1421
      },
      "HP:0002013": {
        "CNS_NAME": "呕吐",
        "ENG_NAME": "Vomiting",
        "LABEL_CAT_ID": 1481
      },
      "HP:0002018": {
        "CNS_NAME": "恶心",
        "ENG_NAME": "Nausea",
        "LABEL_CAT_ID": 1485
      },
      "HP:0002321": {
        "CNS_NAME": "眩晕",
        "ENG_NAME": "Vertigo",
        "LABEL_CAT_ID": 1713
      },
      "HP:0003764": {
        "CNS_NAME": "色素痣",
        "ENG_NAME": "Nevus",
        "LABEL_CAT_ID": 2682
      },
      "HP:0007481": {
        "CNS_NAME": "色素痣",
        "ENG_NAME": "Hyperpigmented nevi",
        "LABEL_CAT_ID": 4832
      },
      "HP:0012531": {
        "CNS_NAME": "疼痛",
        "ENG_NAME": "Pain",
        "LABEL_CAT_ID": 8980
      }
    },
    "POSITION_LIST": [
      [
        22,
        24
      ],
      [
        39,
        41
      ],
      [
        111,
        113
      ],
      [
        129,
        131
      ],
      [
        146,
        148
      ],
      [
        149,
        151
      ],
      [
        208,
        210
      ],
      [
        280,
        282
      ],
      [
        395,
        398
      ],
      [
        430,
        433
      ],
      [
        430,
        433
      ]
    ],
    "TEXT": "现病史：患者于2001左右无明显诱因出现四肢疼痛，以手掌、脚掌为著，天热、感冒发热、运动后加重，休息后可缓解，主要为烧灼痛、胀痛，无明显手脚麻木、感觉减退。并伴有皮肤腰骶、大腿内侧散在零星针尖样大小、紫红色、不高于皮肤表面皮疹，无压痛无瘙痒。近1年前出现两次眩晕，发作时主要为视物旋转，偶伴有恶心，呕吐，无耳鸣等伴随症状，就诊于北京医院诊断为“耳石症”，行复位治疗后好转。近半年患者偶尔有胸闷、憋气、点状针刺样胸前区疼痛症状，发作与体位、活动无明显关联，休息约10-20分钟后可缓解，未曾服用硝酸酯类药物。四肢疼痛较前无明显加重，偶有过电感，无明显麻木。1周前眩晕症状再次出现，外院就诊未予明确诊断。此次为行进一步检查收入我科。 近半年精神、食欲、睡眠可，小便可，近2月大便性状改变，每日2-3次不成形糊状便，颜色不详。体重无明显下降。\n目前诊断：眩晕微循环障碍不除外；听力减退；心肌受累；高血压病（II级 中危组）右侧大脑后动脉纤细；双眼青光眼不除外；右面部色素痣"
  },
  "state": "SUCCESS"
}
```


<h1 id="search-hpo"> search-hpo </h1>

## 接口描述
- 表型检索
- 请求方法：GET

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| query | String | 输入的文本 |

## 输出参数
- Array of [SearchHpoInfo](#SearchHpoInfo)

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/search-hpo?
query=%E5%A4%A7%E5%A4%B4
```

输出示例

```
[
  {
    "CNS_NAME": "大头畸形",
    "CODE": "HP:0000256",
    "ENG_NAME": "Macrocephaly"
  },
  {
    "CNS_NAME": "大头畸形",
    "CODE": "HP:0004482",
    "ENG_NAME": "Relative macrocephaly"
  },
  {
    "CNS_NAME": "出生后大头畸形",
    "CODE": "HP:0005490",
    "ENG_NAME": "Postnatal macrocephaly"
  },
  {
    "CNS_NAME": "大额头",
    "CODE": "HP:0002003",
    "ENG_NAME": "Large forehead"
  },
  {
    "CNS_NAME": "出生时大头畸形",
    "CODE": "HP:0004488",
    "ENG_NAME": "Macrocephaly at birth"
  },
  {
    "CNS_NAME": "头骨的大小异常",
    "CODE": "HP:0000240",
    "ENG_NAME": "Abnormality of skull size"
  }
]
```

<h1 id="search-dis"> search-dis </h1>

## 接口描述
- 疾病检索
- 请求方法：GET
- 注：原来的API为`search-hpo2`

## 输入参数
| 参数名称 | 类型 | 描述 |
|------|----|----|
| query | String | 输入的文本 |

## 输出参数
- Array of [SearchDisInfo](#SearchDisInfo)

## 示例
输入示例：

```
http://www.phenobrain.cs.tsinghua.edu.cn/search-dis?
query=%E5%A4%A7%E5%A4%B4
```

输出示例

```
[
  {
    "CNS_NAME": "腭裂-大耳-小头畸形",
    "CODE": "RD:1731",
    "ENG_NAME": "Cleft palate-large ears-small head syndrome",
    "SOURCE_CODES": [
      "OMIM:181180",
      "ORPHA:2013"
    ]
  },
  {
    "CNS_NAME": "大头畸形/自闭综合征",
    "CODE": "RD:1956",
    "ENG_NAME": "MACROCEPHALY/AUTISM SYNDROME",
    "SOURCE_CODES": [
      "OMIM:605309",
      "ORPHA:210548"
    ]
  },
  {
    "CNS_NAME": "大头-矮小症-截瘫综合征",
    "CODE": "RD:2573",
    "ENG_NAME": "Macrocephaly-short stature-paraplegia syndrome",
    "SOURCE_CODES": [
      "ORPHA:2427"
    ]
  },
  {
    "CNS_NAME": "大头-肥胖症-精神残疾-眼畸形综合征",
    "CODE": "RD:2942",
    "ENG_NAME": "MOMO syndrome",
    "SOURCE_CODES": [
      "OMIM:157980",
      "ORPHA:2563"
    ]
  },
  {
    "CNS_NAME": "X连锁智力残疾-大头-巨睾丸综合征",
    "CODE": "RD:7691",
    "ENG_NAME": "X-linked intellectual disability-macrocephaly-macroorchidism syndrome",
    "SOURCE_CODES": [
      "ORPHA:85320"
    ]
  }
]
```

<h1 id="DataStructure"> 数据结构 </h1>
<h2 id="HpoTreeNode"> HpoTreeNode </h2>

| 参数名称 | 类型 | 描述 |
|------|----|----|
| hpo | String | 表型的HPO编码 |
| CNS_NAME | String | 表型的中文名 |
| ENG_NAME | String | 表型的英文名 |
| children | Array of [HpoTreeNode](#HpoTreeNode) | 子节点列表 |

<h2 id="HpoChildNode"> HpoChildNode </h2>

| 参数名称 | 类型 | 描述 |
|------|----|----|
| CODE | String | 表型的HPO编码 |
| CNS_NAME | String | 表型的中文名 |
| ENG_NAME | String | 表型的英文名 |
| CHILD | Array of String | 表型的子节点HPO编码的列表 |

<h2 id="HpoInfo"> HpoInfo </h2>

| 参数名称 | 类型 | 描述 |
|------|----|----|
| CODE | String | 所查询表型的HPO编码 |
| CNS\_NAME | String | 表型的中文名称 |
| ENG\_NAME | String | 表型的英文名称 |
| CNS\_DEF | String | 表型的中文定义 |
| ENG\_DEF | String | 表型的英文定义 |
| SYNONYM | Array of String | 表型的英文同义词列表 |
| PATH\_DICT | Object | 从根节点到该表型的路径; 当输入中包含TOTAL\_DIS\_NUM则返回该字段 |
| PATH\_INFO\_DICT | Object | 路径上的所有表型节点的信息，当输入中包含TOTAL\_DIS\_NUM则返回该字段 |
| REL\_DIS | Integer | 与该表型关联的罕见病数量 |
| TOTAL\_DIS\_NUM | Integer | 所有罕见病的数目 |

<h2 id="SearchHpoInfo"> SearchHpoInfo </h2>

| 参数名称 | 类型 | 描述 |
|------|----|----|
| CODE | String | 表型的HPO编码 |
| CNS\_NAME | String | 表型的中文名称 |
| ENG\_NAME | String | 表型的英文名称 |


<h2 id="SearchDisInfo"> SearchDisInfo </h2>

| 参数名称 | 类型 | 描述 |
|------|----|----|
| CODE | String | 疾病编码 |
| CNS\_NAME | String | 疾病的中文名称 |
| ENG\_NAME | String | 疾病的英文名称 |
| SOURCE\_CODES | Array of String | 疾病的其他编码列表 |


