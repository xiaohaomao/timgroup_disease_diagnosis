# Phenotype Extraction & Disease Diagnosis

Phenotype Extraction and Differential Diagnosis Pipeline in Diagnosing Rare Diseases Using EHRs





# Brief description of file folders

- `codes/core`：Core package, including data processing, phenotype-based rare disease prediction models

- `codes/bert_syn_project`：Phenotype extraction models 

- `codes/requirement.txt`: List of environment packages for running the code

- `Docker`: docker image to running the code

- `PhenoBrain_Web_API`: Phenobrain website API documentation

  
  
  



# Steps of implementing rare disease differential diagnosis module

#### Implement the following steps to reproduce the main results reported in our paper.



## Software and System Requirements

These are the operating system and software versions used by the authors of the paper. Note that, except for the Python version, other versions are not strictly required and are provided for reference.

Operating System: Ubuntu 22.04.3 LTS

Java: openjdk version 1.8.0_402, required for running the differential diagnosis method BOQA

Python: 3.6.12





## Module 1: Installing basic Python libraries

```
# Create a New Conda Environment，Note that Python version 3.6.12 needs to be 
# installed to avoid potential conflicts with other environment packages.

conda create --name <xxxx> python=3.6.12

# install basic packages based on requirement.txt
pip install -r requirements.txt

```





## Module 2: Download saved models' parameters

Download the trained model and parameters from the following address: https://drive.google.com/drive/folders/1cVApHHw5yLLoLRYZht9Qx52AienJlgWN?usp=sharing. 



Once downloaded, place the model file of the differential diagnosis module in the path '**/codes/core/**'.

The storage model parameters are considerably large, with a total size of approximately 14GB. Our four methods of a size of around 4GB. If you do not care about reproducing the results of 12 baseline methods. In that case, you are recommended to download our models solely from the following five file folders: **ICTODQAcrossModel, HPOICCalculator, HPOProbMNBNModel, LRNeuronModel, CNBModel**.





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

Second, put the releated patient data in this folder "**codes/core/data/preprocess/patient/**".

The main test datasets have been publicly released at

[Zenodo]: https://zenodo.org/records/10774650)	"Zenodo."

### Step 3

To reproduce all the results discussed in the supplementary file of this study, run the "**core/script/test/test_optimal_model.py**" file.

```
# Running an Example

python core/script/test/test_optimal_model.py
```



The main file that contains all the settings for running the model and generating results is "**core/script/test/test_optimal_model.py**".



In "**core/helper/data/data_helper.py**", you can find all the addresses of the datasets used in this study. You can replace them with your own if you want to use a different address.



The "**core/predict/**" folder comprehensively describes rare disease diagnosis models. In addition to the 17 methods (12 state-of-the-art baselines "**core/predict/sim_model/**" and our 4 developed or used methods, ICTO,PPO,CNB,MLP), After obtaining the results from the 4 models, a new disease prediction ranking can be generated by combining the prediction results of the four models using the ensemble method based on order statistics. 



**The Ensemble model**. We observed that a single assumption or model would not completely capture the characteristics of the diverse datasets. Hence, we developed the Ensemble model by combining predictions from multiple methods using order statistics, and it achieved better results. The Ensemble model calculates one overall prediction by integrating the rankings of the previous four methods using order statistics. Suppose the number of methods is N . First, the Ensemble method normalizes the ranking of diseases within each method to obtain ranking ratios. It then calculates a Z statistic, which measures the likelihood that the observed ranking ratios are solely due to chance factors. It calculates the probability of obtaining ranking ratios through random factors that are smaller than the currently observed ranking ratios. Under the null hypothesis, the position of each disease in the overall ranking is random. In other words, for two diseases, the one with a smaller   statistic is more likely to have a top rank. The joint cumulative distribution of an N-dimensional order statistic is used to calculate the Z statistics:
$$
Z(r_1,r_2,...,r_N)=N!\int_0^{r_1}\int_{s_1}^{r_2}...\int_{s_{N-1}}^{r_N}ds_N ds_{N-1} ...ds_1,
$$
where  ri is the rank ratio by the  i-th method, and r0=0 . Due to its high complexity, we implemented a faster recursive formula to compute the above integral as previously done：
$$
V_{k}=\sum_{i=1}^{k}(-1)^{i-1}\frac{V_{k-i}}{i!}r_{N-K+1}^{i},\\Z(r_{1},r_{2},...,r_{N})=N! V_{N},
$$
where  V0=1, and  ri is the rank ratio by the  i-th method. 

For more information, please refer to the reference [1] .

Moreover, it contains more deep learning models, GCN, and Bayesian network. Available for user to try.





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

1. Go to the Diagnose interface.
2. Select the diagnostic method and the number of predicted results to show.
3. Finally, press the **“Predict”** button to get the prediction results for each method within a few seconds.





**Example of Disease Diagnosis** 

!["Example of disease diagnosis"](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/example_result/disease%20diagnosis.png)



# Steps of implementing phenotype extraction module

TBC



## References

1. Aerts S, Lambrechts D, Maity S, Van Loo P, Coessens B, De Smet F, Tranchevent L-C, De Moor B, Marynen P, Hassan B. Gene prioritization through genomic data fusion. *Nature biotechnology*. 2006;**24**.

   







