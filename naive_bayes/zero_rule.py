from util import labels_to_key, parse_federalist_papers, labels_to_y, find_zero_rule_class, \
    apply_zero_rule
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse

def main(data_file, random_seed):
    print(data_file)

    # TODO: For this lab, you do NOT need to make any changes in this file, only util.py
    # Running this script checks that they're defined correctly
    # TODO 1: load the data by defining parse_federalist_papers in util
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    num_essays = len(essays)
    print(f"1: Working with {num_essays} reviews")

    # TODO 2: create a key that links author id string -> integer by defining labels_to_key in util
    author_key = labels_to_key(authors)
    print(f"2: Author key {author_key}")

    # TODO 3: convert the list of strings in `authors` to a np array by defining labels_to_key in util
    # convert all the labels using the key
    y = labels_to_y(authors, author_key)
    assert y.size == len(authors), f"Size of label array (y.size) must equal number of labels {len(authors)}"

    # shuffle and split the data. Function is defined in sklearn.model_selection
    train_X, test_X, train_y, test_y = train_test_split(essays, y, test_size = 0.3, random_state = random_seed)
    print(f"{len(train_X)} in train; {len(test_X)} in test")

    # TODO 4: define find_zero_rule and apply_zero_rule in util
    # the code below learns the zero rule on train data and then uses it to classify held-out essays
    most_frequent_class = find_zero_rule_class(train_y)
    # lookup label string from class #
    reverse_author_key = {v:k for k,v in author_key.items()}
    print(f"2. The most frequent class is {reverse_author_key[most_frequent_class]}")

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(test_X, most_frequent_class)
    print(f"3. Zero rule predictions on held-out data: {test_predictions}")

    # Report the accuracy of zero-rule predictions
    test_accuracy = accuracy_score(test_y,test_predictions)
    print(f"4. Accuracy of zero rule: {test_accuracy:0.03f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test supervised learning utilities')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed for dataset split')
    args = parser.parse_args()

    main(args.path, args.seed)
