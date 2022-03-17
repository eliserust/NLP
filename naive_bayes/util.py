import numpy as np
from collections import Counter
import json



def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

# TODO: lab 1

def parse_federalist_papers(data_file):
    # open the file
    # go through the items in the file
    # append a string to each list
    authors = []
    texts = []
    essay_ids = []
    with open(data_file, 'r') as df:
        data = json.load(df)
        for item in data: # loop through each json element
            author, essay, text = item # A list of lists in the json file so easily sectioned
            authors.append(author)
            texts.append(text)
            essay_ids.append(essay)
    return authors, texts, essay_ids


# TODO: write this function (lab1, homework)
def labels_to_key(labels): # labels is a list of strings
    """
    Creates a mapping from string representations of labels (hamilton) to integers (0 or 1)
    :param labels:
    :return: label_key, dict {str: int}
    """
    label_set = set(labels) # transform list of strings to set (only care about unique ones)
    label_key = {} # create key to hold dictionary info
    for i, label in enumerate(label_set): # enumerate set, get 0, Madison out of that so you can then assign to dictionary
        #print(i)
        #print(label)
        label_key[label] = i
    return label_key

# TODO: write this function (lab1, homework)
def labels_to_y(labels, label_key):
    """
    :param labels: list of strings
    :param label_key: dictionary {str: int}
    :return: numpy vector y
    """
    y = np.zeros(len(labels), dtype=np.int)
    for i,l in enumerate(labels):
        y[i] = label_key[l]
    return y

# TODO: write this function (lab1, homework)
def find_zero_rule_class(train_y):
    # what is the most common element in this array?
    # use a Counter, get max based on highest counter value
    """
    Determines the class predicted by the zero rule algorithm
    :param train_y: training labels
    :return: most_freq, the most frequent element in train_y
    """
    class_counts = Counter(train_y)
    print(class_counts)
    most_freq = max(class_counts, key = lambda k: class_counts[k])
    return most_freq

# TODO: write this function (lab1, homework)
def apply_zero_rule(X, zero_class):
    """
    Predicts most frequent class using zero rule algorithm
    :param X: iterable, data to classify
    :param zero_class: class to predict
    :return: classifications: numpy array
    """
    classifications = np.zeros(len(X), dtype=np.int)
    # assign every y the zero class
    classifications[:] = zero_class
    return classifications
