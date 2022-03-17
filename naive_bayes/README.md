# Naive Bayes for authorship attribution

# Experimental results
TODO: Fill tables and write your description of this experiment here (see Homework description below)

The experiment uses federalist_dev.json to classify different Federalist Papers according to their authors. The two authors in the data are James Madison and Alexander Hamilton - two of the most prolific authors of the 85 essays in the Federalist Papers. Experts estimate that Hamilton wrote 51 of the 85 essays, while Madison wrote 29 of the essays. Thus, we use different Naive Bayes Models to do this classification. We take the raw essays and generate a matrix of feature counts (how many times a feature - or word- appears in each essay). We then divide this matrix and a vector of author names (as 1s and 0s) into training and testing sets. These are used to fit a Multinomial Naive Bayes and a Bernoulli Naive Bayes model to compare against the Zero Rule model. As shown below, the data is partitioned into 25% test (17 essays) and 75% train (49) essays and the Zero-Rule model predicts author labels accurately 64.7% of the time. Bernoulli Naive Bayes sees a higher prediction accuracy at 76.47% of the time - due to the ability of the model to learn which authors use which words (with binary features). However, by far the most accurate model is Multinomial Naive Bayes which intakes a matrix of feature counts for each essay (and thus for each author) and predicts authorship accurately 94.1% of the time. 


Data Partition | Num Essays
---------------| ----------
Train | 49
Test | 17

Model | Accuracy (test)
---------------| ----------
MultinomialNB | 0.9411764705882353
BernoulliNB | 0.7647058823529411
Zero-rule | 0.647



# Files

## `federalist_dev.json` and `federalist_test.json`

.json files containing the text of federalist papers written by Madison, Hamilton, or disputed between the two authors.
The dev file is all labeled. It can be split and used for development (training and validation). 

The test file contains only the disputed papers and no labels. It is not used in the labs and homework.

## Lab, week 1 : `zero_rule.py`

`util.py` implements functions to load data and get the zero-rule baseline for supervised learning.
You will not have to make changes to the script `zero_rule.py`, only `util.py` to complete this lab.

The functions are 
* creating labels as numpy arrays
* implementing the zero-rule algorithm as a baseline / scoring accuracy

Usage: `python zero_rule.py --path federalist_dev.json`

## Lab, week 2 : `handmade_nb.py`

This "handmade" Naive Bayes lab implements the simple math of the model!

Usage: `python handmade_nb.py --path federalist_dev.json`

## Homework, week 2 : `sklearn_nb.py`

Usage: `python sklearn_nb.py --function_words_path ewl_function_words.txt --path federalist_dev.json`

Apply `sklearn.naive_bayes.MultinomialNB` and `BernoulliNB` to two authors, as defined in the starter code. 
Consult the scikit learn docs to better understand how to interact with this model.

Refer to the feature extraction homework to create data in the right format for this model: 
concatenate all feature vectors to create input matrix X; create a label vector y.

Assign the labeled data to train and test sets: 75% train and 25% test. 
Use this same split for all experiments in the script.

Fit and evaluate three models; our metric is accuracy:
* zero-rule baseline
* Multinomial Naive Bayes with count features
* Bernoulli Naive Bayes with binary features

Update the Experimental Results section at the top of this README 
with a brief summary (~8 sentences) of the dataset, methods and results (i.e. model accuracy), 
comparing the test results on your two models and the baseline. 
Include which author is predicted by the zero rule baseline.

_Naive Bayes is *deterministic*, meaning the model's probability estimates are always the same, given the same inputs. 
If you pass different random seeds to the dataset splitter, you will see different results as different essays
as assigned to the train and validation partitions._


