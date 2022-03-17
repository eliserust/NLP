#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, labels_to_key, labels_to_y, find_zero_rule_class, apply_zero_rule
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.utils import shuffle
from nltk import word_tokenize
import sys

def load_features(list_of_essays, list_of_features):
    X = np.zeros((len(list_of_essays), len(list_of_features)), dtype=int)
    for i, essay in enumerate(list_of_essays): # loop through all essays
        tokens = word_tokenize(essay.lower())
        #print(tokens)
        for j, feature in enumerate(list_of_features):
            count = len([token for token in tokens if token == feature]) # list comprehension
            X[i,j] = count
    return X


def main(data_file, vocab_path, random_seed):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""

    function_words = load_function_words(vocab_path)
    authors, essays, essay_ids = parse_federalist_papers(data_file)


    # TODO 1. define the function above to load attributed essays into a feature matrix
    X = load_features(essays, function_words)
    print(f"Numpy feature array has shape {X.shape} and dtype {X.dtype}")

    # TODO 2: load the author names into a vector y, mapped to 0 and 1, using functions from util
    labels_map = labels_to_key(authors)
    y = np.asarray(labels_to_y(authors, labels_map))
    print(y)


    # TODO 3: shuffle, then split the attributed data using util. Assign 75% train / 25% test
    # Use the train_test_split function in sklearn.model_selection and make sure to pass random_seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    print(f"The training data has shape {X_train.shape} and dtype {X_train.dtype}")
    print(f"The testing data has shape {X_test.shape} and dtype {X_test.dtype}")
    print(f"The training labels have shape {y_train.shape} and dtype {y_train.dtype}")
    print(f"The testing labels has shape {y_test.shape} and dtype {y_test.dtype}")

    # (TODO 3-6 will be evaluated by checking object/function use in code and via writeup in README
    # (..., I suggest using print statements to get info for the writeup)

    # TODO 4: train a multinomial NB model, evaluate on validation split
    multi_model = MultinomialNB()
    multi_model.fit(X_train, y_train) # Need to make y_train (labels)

    # Test on validation split
    test_multi = multi_model.score(X_test, y_test) # Compute accuracy --> .score()

    print("\nThe prediction accuracy from Multinomial NB is:")
    print(test_multi)


    # TODO 5: train a Bernoulli NB model, evaluate on validation split
    X_train_binary = X_train != 0 # convert X_train to binary matrix
    X_test_binary = X_test != 0 # convert X_test to binary matrix
    #print(X_train_binary)

    bernoulli_model = BernoulliNB()
    bernoulli_model.fit(X_train_binary, y_train)  # Need to make y_train (labels)

    # Test on validation split
    test_bern = bernoulli_model.score(X_test_binary, y_test)

    print("\nThe prediction accuracy from Bernoulli NB is:")
    print(test_bern)

    # TODO 6: fit the zero rule, evaluate on validation split
    # learns the zero rule on train data
    most_frequent_class = find_zero_rule_class(y_train)
    #print(most_frequent_class)

    # lookup label string from class #
    reverse_author_key = {v: k for k, v in labels_map.items()}
    print(f"2. The most frequent class is {reverse_author_key[most_frequent_class]}")

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(X_test, most_frequent_class)
    print(f"3. Zero rule predictions on held-out data: {test_predictions}")

    # Report the accuracy of zero-rule predictions
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"4. Accuracy of zero rule: {test_accuracy:0.03f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed for dataset split')
    args = parser.parse_args()

    main(args.path, args.function_words_path, args.seed)

    # sys.exit()