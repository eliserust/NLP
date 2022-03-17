import argparse
import numpy as np
from util import parse_sts, sts_to_pi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys

def main(sts_data, max_nonparaphrase, min_paraphrase, cos_sim_threshold):
    """Transform a semantic textual similarity dataset into a paraphrase identification.
    Data is formatted as in the STS benchmark"""

    max_nonparaphrase = 3.0
    min_paraphrase = 4.0

    # TODO 1: define parse_sts in util.py to read sts benchmark data
    texts, labels = parse_sts(sts_data)


    # TODO 2: Define this function in util.py that converts a STS labels to binary paraphrase labels
    # since it removes items between max_nonparaphrase and min_paraphrase, you will have fewer items after
    pi_texts, pi_labels = sts_to_pi(texts, labels, max_nonparaphrase, min_paraphrase)

    print(pi_texts)
    print(pi_labels)


    # TODO 3: Check that you have the right number of items after converting to paraphrase labels
    # some items will be discarded because they are between the threshholds
    non_in_sts_labels = len([label for label in labels if label <= max_nonparaphrase])
    para_in_sts_labels = len([label for label in labels if label >=min_paraphrase])

    num_nonparaphrase = (pi_labels == 0).sum()
    num_paraphrase = (pi_labels == 1).sum()
    # for dev, 957 and 264
    assert num_nonparaphrase == non_in_sts_labels, f"Found {num_nonparaphrase} non-paraphrase items; does not match expected"
    assert num_paraphrase == para_in_sts_labels, f"Found {num_paraphrase} paraphrase items; does not match expected"

    # TODO 4: This can be mostly copied from last lab
    all_t1, all_t2 = zip(*pi_texts)
    all_texts = all_t1 + all_t2

    # Instantiate a TFIDFVectorizer to create representations for sentences
    vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True, min_df = 10)
    vectorizer.fit(all_texts)

    # Use word 1-grams, but your choice what further preprocessing you apply
    # Compute cosine similarity for each pair of sentences
    cos_sims = []
    for pair in pi_texts:
        # each item is a 2-tuple
        # this menas we will get a (2, |vocab|) sparse representation back
        pair_reprs = vectorizer.transform(pair)
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        cos_sims.append(pair_similarity[0,0])

    # Uses a threshold specified by command line args to convert each similarity score into a paraphrase prediction
    predictions = np.asarray(cos_sims) > cos_sim_threshold


    # TODO 5: calculate and print precision and recall statistics for your system
    num_pred = predictions.sum()
    print(f"Number predicted paraphrase: {num_pred}")

    pi_labels = np.asarray(pi_labels)
    num_pos = pi_labels.sum()
    print(f"Number positive: {num_pos}")

    true_pos = ((pi_labels * 1 + predictions * 1) == 2)
    num_true_pos = true_pos.sum()
    print(f"Number true positive: {num_true_pos}")

    precision = num_true_pos / num_pred
    recall = num_true_pos / num_pos
    print(f"Scores: precision {precision:0.03}\trecall {recall:0.03}")

    # double check our work: use sklearn's implementation
    p = precision_score(pi_labels, predictions)
    r = recall_score(pi_labels, predictions)
    a = accuracy_score(pi_labels, predictions)
    f1 = f1_score(pi_labels, predictions)
    print(f"Sklearn scores: accuracy {a:0.03}\tprecision {p:0.03}\trecall {r:0.03}\tf1 {f1:0.03}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="../paraphrase-identification-eliserust/stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    parser.add_argument("--max_nonparaphrase", type=float, default=3.0)
    parser.add_argument("--min_paraphrase", type=float, default=4.0)
    parser.add_argument("--cos_sim_threshold", type=float, default=0.7)
    args = parser.parse_args()

    main(args.sts_data, args.max_nonparaphrase, args.min_paraphrase, args.cos_sim_threshold)

# Command line
# conda activate pycomplx
# python threshold_baseline.py --cos_sim_threshold 0.5