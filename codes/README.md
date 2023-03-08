# Description
- `core`：Core package, including data processing, phenotype-based rare disease prediction models, and experiments

- `bert_syn_project`：Phenotype extraction models and experiments

  



# Steps of implementing rare disease diagnosis algorithm

#### Implement the following steps to reproduce the results in our paper.



### Step 1: Install requirements



```
pip install -r requirements.txt
```

then, install TensorFlow-GPU=1.14.0



### Step 2: Download saved models' parameters

Download  trained model and parameters from the following address.

https://drive.google.com/drive/folders/1EKn73BPx2DJQnF6ApQYwrK41gmVD2ndd?usp=sharing



Then, put the model file of differential diagnosis module in the path "[Rare-Diseases-PhenoBrain](https://github.com/xiaohaomao/Rare-Diseases-PhenoBrain)/codes/core/".





### Step 3: Implement differential diagnosis models 

```
RAREDIS_PATH = your address + "Rare-Diseases-PhenoBrain/codes/core"

CORE_PATH="${RAREDIS_PATH}/core"
export PYTHONPATH=$CORE_PATH:$PYTHONPATH
cd $CORE_PATH

#  run the following codes to generate results
python core/script/test/test_optimal_model.py
```





### 

