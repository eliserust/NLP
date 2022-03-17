import argparse
import numpy as np
from util import parse_sts, sts_to_pi
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import edit_distance
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sys


def load_X(sent_pairs, tfidf_vectorizer):
    """Create a matrix where every row is a pair of sentences and every column is a feature.
    Feature (column) order is not important to the algorithm."""


    features = ["BLEU", "Word Error Rate", "Tfidf"]
    #X = np.zeros((len(sent_pairs), len(features))) # initialize matrix X

    # BLEU
    bleu_scores = []
    for text_pair in sent_pairs:
        t1, t2 = text_pair
        t1_tokens = word_tokenize(t1.lower())
        t2_tokens = word_tokenize(t2.lower())
        bleu_smoothing = SmoothingFunction().method4
        score = sentence_bleu([t1_tokens, ], t2_tokens, smoothing_function = bleu_smoothing)
        bleu_scores.append(score)
    #print(bleu_scores)

    # Word Error Rate
    wer_scores = []
    for text_pair in sent_pairs:
        t1, t2 = text_pair
        t1_tokens = word_tokenize(t1.lower())
        t2_tokens = word_tokenize(t2.lower())
        wer = edit_distance(t1_tokens, t2_tokens)
        wer_rate = wer / (len(t1) + len(t2))
        wer_scores.append(wer_rate)
    #print(wer_scores)

    # TFIDF - Cosine Similarity
    cos_sims = []
    for pair in sent_pairs:
        # each item is a 2-tuple
        # this menas we will get a (2, |vocab|) sparse representation back
        pair_reprs = tfidf_vectorizer.transform(list(pair))
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        cos_sims.append(pair_similarity[0, 0])
    #print(cos_sims)

    # turn into array
    X = np.array([bleu_scores, wer_scores, cos_sims]).T

    return X


def main(sts_train_file, sts_dev_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    min_paraphrase = 4.0
    max_nonparaphrase = 3.0

    # TODO 1: Load data partitions and convert to paraphrase dataset as in the lab
    # You will train a logistic regression on the TRAIN parition
    train_texts_sts, train_y_sts = parse_sts(sts_train_file) # parse from training file
    train_texts_pi, train_y_pi = sts_to_pi(train_texts_sts, train_y_sts, max_nonparaphrase, min_paraphrase) # convert to PI
    #print(type(train_texts_pi))

    # You will evaluate predictions on the VALIDATION partition
    dev_texts_sts, dev_y_sts = parse_sts(sts_dev_file)
    dev_texts_pi, dev_y_pi = sts_to_pi(dev_texts_sts, dev_y_sts, max_nonparaphrase, min_paraphrase)

    # TODO 2: Train a logistic regression model using sklearn.linear_model.LogisticRegression
    # Hint: The interface is very similar to other sklearn models we have used in class

    # Load_X for train
    all_t1, all_t2 = zip(*train_texts_pi)
    all_texts = all_t1 + all_t2

    tfidf_vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True, min_df = 10)
    tfidf_vectorizer.fit(all_texts)
    train_X = load_X(train_texts_pi, tfidf_vectorizer)
    #print(train_X)


    # Load_X for dev
    dev_X = load_X(dev_texts_pi, tfidf_vectorizer)
    #print(dev_X)

    # Logistic Regression
    logistic_reg = LogisticRegression() # initialize
    logistic_reg.fit(train_X, train_y_pi)

    # TODO 3: Evaluate your logistic regression model on dev using accuracy, precision, recall and F1
    # Get predictions for the dev partition to do this
    log_predictions = logistic_reg.predict(dev_X)
    print(log_predictions)


    # TODO 4: You will need to report the same evaluation metrics (accuracy, precision, etc.) for the baseline in the lab
    # You may choose to do it in this script or to add more evaluation to the end of your lab script
    a = accuracy_score(dev_y_pi, log_predictions)
    p = precision_score(dev_y_pi, log_predictions)
    r = recall_score(dev_y_pi, log_predictions)
    f1 = f1_score(dev_y_pi, log_predictions)
    print(f"Sklearn scores: accuracy {a:0.03}\tprecision {p:0.03}\trecall {r:0.03}\tf1 {f1:0.03}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="../paraphrase-identification-eliserust/stsbenchmark/sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="../paraphrase-identification-eliserust/stsbenchmark/sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)
