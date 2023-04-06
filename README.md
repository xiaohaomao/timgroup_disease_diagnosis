# Disease Diagnosis

Differential Diagnosis Pipeline in Diagnosing Rare Diseases Using EHRs





# Brief description of two file folders

- `codes/core`：Core package, including data processing, phenotype-based rare disease prediction models

- `codes/bert_syn_project`：Phenotype extraction models 

  



# Steps of implementing rare disease diagnosis models

#### Implement the following steps to reproduce the main results reported in our paper.



## Module 1: Installing basic Python libraries

```
# install basic packages
pip install -r requirements.txt

# Install the GPU version of TensorFlow with 1.14.0
pip install tensorflow-gpu==1.14.0
```





## Module 2: Download saved models' parameters

Download the trained model and parameters from the following address: https://drive.google.com/drive/folders/1cVApHHw5yLLoLRYZht9Qx52AienJlgWN?usp=sharing. 



Once downloaded, place the model file of the differential diagnosis module in the path '**/codes/core/**'.

The storage model parameters are considerably large, with a total size of approximately 14GB. Our five methods of a size of around 4GB. If you do not care about reproducing the results of 12 baseline methods. In that case, you are recommended to download our models solely from the following five file folders: **ICTODQAcrossModel, HPOICCalculator, HPOProbMNBNModel, LRNeuronModel, CNBModel**.





##  Module 3: Run differential diagnosis models to generate results in this study

### step 1

First, create a new terminal window, and set the default environment by following the commands.

```
#  codes in your own address
CODE_PATH= your address+ "timgroup_disease_diagnosis"

CORE_PATH="${CODE_PATH}/core"
export PYTHONPATH=$CORE_PATH:$PYTHONPATH
cd $CORE_PATH
```



### Step 2

Second, place the releated patient data in this folder "**core/data/**".



### Step 3

To reproduce all the results discussed in the supplementary file of this study, run the "**core/script/test/test_optimal_model.py**" file.

```
# Running an Example

python core/script/test/test_optimal_model.py
```



The main file that contains all the settings for running the model and generating results is "**core/script/test/test_optimal_model.py**".



In "**core/helper/data/data_helper.py**", you can find all the addresses of the datasets used in this study. You can replace them with your own if you want to use a different address.



The "**core/predict/**" folder comprehensively describes rare disease diagnosis models. In addition to the 17 methods (12 state-of-the-art baselines and our 5 developed or used methods), it contains more deep learning models, GCN, and Bayesian network.





## Module 4: Results illustrate

After running the test_optimal_model.py file, the program will generate seven folders, namely **CaseResult, csv, delete, DisCategoryResult, Metric-test, RawResults and table**.

The **CaseResult** folder encompasses a comprehensive list of predicted diseases for all patients, out of a total of 9260 diseases, for each of the methods employed. Sample results for PUMCH-ADM and validation of RAMEDIS are included below. 

![**Example: Each diagnostic method's predicted ranking for each patient's datad**](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/example_prediction_each_case.png)



In the **table** folder, a comparison of multiple statistical metrics is available for each method applied to every dataset. Sample results for PUMCH-ADM and validation of RAMEDIS are included below. These metrics provide a detailed insight into the performance of each method applied to the dataset analyzed.

![**Example: Multiple statistical metrics of each diagnostic method on PUMCH-ADM and the validation subset of Ramedis dataset**](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/example_validation_ramedis-pumcha_adm.png)



The **RawResults** folder comprises complete ranked lists of predicted diseases for each method implemented on every patient of each dataset, covering a total of 9260 diseases. The raw predictions saved in this folder can range from a few MB to several GB in size. The user has the option to choose whether or not to store these raw predictions by altering the settings in the **test_optimal_model.py** file.



## Module 5: Diagnostic tool illustrate

We upload the trained model code to PhenoBrain, which contains two modules, Phenotype Extraction and Differential Diagnosis/Disease Prediction. Users can select phenotypes in three ways: by phenotype tree for precise phenotype selection, by the phenotype search function, and by the phenotype extraction function. The phenotype extraction module is in the input interface. Users enter clinical text into the phenotype extraction box, select the method of phenotype extraction **(HPO/CHPO, CHPO-UMLS, CText2Hpo)**, and press the **"Extract"** button to extract the phenotypes. The interface's right side presents the extracted phenotype's specific information. 



!["Interface of phenotype_extraction"](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/phenotype_extraction.png)





Specifically, we prepared three examples in PhenoBrain to verify the effect of the phenotype extraction function. The input is Chinese clinical text, English clinical text, and the HPO code list. Then press the **"Extract"** key to demonstrate the effect of extracting phenotype.







**Example of Phenotype Extraction in English Text**

!["Example of extracting phenotype in English"](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/phenotype_extract_example_english.png)







**Example of Phenotype Extraction in Chinese Text**

!["Example of extracting phenotype in Chinese"](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/phenotype_extraction_example_chinese.png)





**After selecting the phenotype:**

​	 **1，**Go to the Diagnose interface.

 	**2**，Select the diagnostic method and the number of predicted results to display.

​	 **3，**Finally, press the **“Predict”** button to get the prediction results for each method within a few seconds.





**Example of Disease Diagnosis** 

!["Example of disease diagnosis"](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/disease%20diagnosis.png)



















