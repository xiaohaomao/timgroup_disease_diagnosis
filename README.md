# Disease Diagnosis

Differential Diagnosis Pipeline in Diagnosing Rare Diseases Using EHRs





# Brief description of two file folders

- `codes/core`：Core package, including data processing, phenotype-based rare disease prediction models

- `codes/bert_syn_project`：Phenotype extraction models 

  



# Steps of implementing rare disease diagnosis algorithm

#### Implement the following steps to reproduce the main results reported in our paper.



### Module 1: Installing basic Python libraries

```
# install basic packages
pip install -r requirements.txt

# Install the GPU version of TensorFlow with 1.14.0
pip install tensorflow-gpu==1.14.0
```





### Module 2: Download saved models' parameters

Download the trained model and parameters from the following address: https://drive.google.com/drive/folders/1cVApHHw5yLLoLRYZht9Qx52AienJlgWN?usp=sharing. 



Once downloaded, place the model file of the differential diagnosis module in the path '**/codes/core/**'.

The storage model parameters are considerably large, with a total size of approximately 14GB. Our five methods of a size of around 4GB. If you do not care about reproducing the results of 12 baseline methods. In that case, you are recommended to download our models solely from the following five file folders: **ICTODQAcrossModel, HPOICCalculator, HPOProbMNBNModel, LRNeuronModel, CNBModel**.





###  Module 3: Run differential diagnosis models to generate results in this study

#### step 1

First, create a new terminal window, and set the default environment by following the commands.

```
#  codes in your own address
CODE_PATH= your address+ "timgroup_disease_diagnosis"

CORE_PATH="${CODE_PATH}/core"
export PYTHONPATH=$CORE_PATH:$PYTHONPATH
cd $CORE_PATH
```



#### Step 2

Second, place the releated patient data in this folder "**core/data/**".



#### Step 3

To reproduce all the results discussed in the supplementary file of this study, run the "**core/script/test/test_optimal_model.py**" file.

```
# Running an Example

python core/script/test/test_optimal_model.py
```



The main file that contains all the settings for running the model and generating results is "**core/script/test/test_optimal_model.py**".



In "**core/helper/data/data_helper.py**", you can find all the addresses of the datasets used in this study. You can replace them with your own if you want to use a different address.



The "**core/predict/**" folder comprehensively describes rare disease diagnosis models. In addition to the 17 methods (12 state-of-the-art baselines and our 5 developed or used methods), it contains more deep learning models, GCN, and Bayesian network.







### Module 4: Results Display

The program will generate 7 folders after running the **test_optimal_model.py file:** **CaseResult, csv, delete, DisCategoryResult, Metric-test, RawResults, table**. 

The **CaseResult** folder contains a ranking of predicted diseases for all patients among a total of 9260 diseases for each method used, with examples of the results (PUMCH-ADM, the validation of RAMEDIS) provided below.

![](/Users/maoxiaohao/Desktop/投Nature Medicine 草稿/Submitted_version/送外审版本/NM_manuscript/methods_codes/timgroup_disease_diagnosis/example_result/example_prediction_each_case.png)



Within the **table** folder, a comparison of multiple statistical metrics is available for each method applied to every dataset, with examples of the results (PUMCH-ADM, the validation of RAMEDIS) provided below.

![alt](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/example_prediction_each_case.png)

Within the **RawResults** folder, each method applied to every patient of each dataset has a complete ranked list of predicted diseases, out of a total of 9260. The raw predictions stored for each method range from a few MB to several GB. The user can choose whether or not to store these raw predictions in the **test_optimal_model.py** file





