Paraphrase Identification using string similarity
---------------------------------------------------

This project examines string similarity metrics for paraphrase identification (PI).
It converts semantic textual similarity data to paraphrase identification data using threshholds.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting paraphrase.


Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

## Results

TODO: Write up your results for the baseline implemented in the lab and the logistic regression model in the homework:

* Describe the dataset (1 sentence),

The dataset is the STS benchmark dataset and companion dataset, which is a collection of 8628 sentence pairs and their associated (human deemed) similarity score. Each sentence pair varies in its paraphrase status - with some pairs highly semantically similar and others almost completely different in meaning.

* describe the baseline (including what threshold you used) and the logistic regression (2+ sentences each), 

The baseline evaluates cosine similarity of TFIDF vectors at a threshold of 0.7. Any similarity score above this threshold is converted into a paraphrase prediction of True, and any below are predicted as False.
The logistic regression uses multiple similarity scores (Cosine Similarity, BLEU, and Word Error Rate) to train an sklearn Logistic Regression model on the metrics. Logistic regression in paraphrase identification identifies a "decision boundary" on which every new sentence pair must be classified as a paraphrase or non-paraphrase.

* fill the table with evaluation on the dev partition,
* and compare the results (3 sentences).

Precise results will vary a little based on preprocessing choices.

| Model Name | Accuracy | Precision | Recall | F1|
| ---------- | -------- | --------- | ------- | ---|
| Baseline   |0.4       |   0.589   | 0.564   | 0.576|
| Logistic Regression   |0.838      |   0.67   | 0.492   | 0.568|

The logistic regression model clearly outperforms the baseline across the board. The most drastic improvement is in the Accuracy score (0.434 points higher for log regression) - though this is maybe the least important validation measure - while the smallest improvement is in F1 (0.004 points higher for log regression). On the whole, however, recall and F1 scores were fairly low for both models (0.53 - 0.58) and the Logistic Regression model could be improved with different preprocessing choices or different similarity scores.

## Homework: pi_logreg.py

* Train a logistic regression for paraphrase identification on the training data using three features:
    - BLEU
    - Word Error Rate
    - Cosine Similarity of TFIDF vectors
* Use the logistic regression implementation in `sklearn`.
* Update the readme as described in *Results*.

`python pi_logreg.py --sts_dev_file stsbenchmark/sts-dev.csv --sts_train_file stsbenchmark/sts-train.csv`

## Lab: threshold_baseline.py

`threshold_baseline.py` converts a STS dataset to paraphrase identification
 and checks the distribution of paraphrase/nonparaphrase.
Then, it evaluates TFIDF vector similarity as a model of paraphrase by setting a threshold and
considering all sentence pairs above that that similarity to be paraphrase.

Example usage:

`python threshold_baseline.py --sts_data stsbenchmark/sts-dev.csv --cos_sim_threshold 0.8`


