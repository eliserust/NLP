import numpy as np

# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """

    texts = []
    labels = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5]
            t2 = fields[6]
            texts.append((t1, t2))

    return texts, labels


# TODO: lab, homework
def sts_to_pi(texts, sts_labels, max_nonparaphrase, min_paraphrase):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase."""

    # loop through sts labels, get index if the label is below min or above max
    pi_rows = [i for i, label in enumerate(sts_labels)
               if label >= min_paraphrase or label <= max_nonparaphrase]
    pi_texts = [texts[i] for i in pi_rows] # get text from those rows

    # using indexing to get the right rows out of labels
    pi_labels = np.asarray(sts_labels)[pi_rows]
    # convert to binary using threshold
    pi_labels = pi_labels > max_nonparaphrase

    return pi_texts, pi_labels
