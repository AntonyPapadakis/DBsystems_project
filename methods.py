import numpy
import pandas as pd
import re
from nltk.util import ngrams
from collections import Counter
import numpy as np


def map_to_vector(V, s):
    """
    :param V: vocabulary
    :param s: list of ngrams for a certain query
    :return: the representation vector which is a sum of all of the onehot encoded vectors of each ngram in s
    with respect to the vocabulary V
    """

    representation_vector = np.zeros(len(V))

    for t in s:
        if t in V:
            onehot_vector = np.zeros(len(V))
            i = V.index(t)
            onehot_vector[i] = 1
            representation_vector = np.add(representation_vector, onehot_vector)

    return representation_vector


def ngrams_from_query(n: int, q: str, result: list):
    q = q.lower()
    q = re.sub(r'\b[0-9]+\b|\s', ' ',
               q)  # regular expression to replace all digits and whitespace chars but not alphanumerics
    tokens = [token for token in q.split(" ") if token != ""]

    for num_of_ngrams in range(2, n + 1):  # we find the most frequent ngrams up to the given n
        # getting the ngrams and storing them
        output = list(ngrams(tokens, num_of_ngrams))  # trigrams
        result.extend(output)
    return result


def bag_of_ngrams_Vocabulary(n: int, data: pd.DataFrame) -> (list, list):
    result = []
    for q in data.statement:
        result = ngrams_from_query(n, q, result)

    out = Counter(result)
    count_items = 0
    Vocabulary = []
    V_counter = []

    for key, value in out.items():  # iterating over  the dictionaries' items

        if value >= 300:  # appending the most frequent ngrams to the vocabulary
            Vocabulary.append(key)
            V_counter.append(value)
            count_items += 1

    print(count_items, "\n", Vocabulary)

    return Vocabulary, V_counter


def bag_of_ngrams_get_vector(n: int, data: pd.DataFrame, vocabulary: list) -> (list, list):
    # for every query calculate the ngrams and their one hot encoding with
    # respect to each ngram in the Vocabulary
    # and then add them to create the vector representation for each query

    # declaring list of representation vectors
    # queries are mapped to u dimmensional vectors
    u = []

    count = 0
    for q in data.statement:
        result = []
        result = ngrams_from_query(n, q, result)
        r_vector = map_to_vector(vocabulary, result)
        u.append(r_vector)
        count += 1

    total_tokens_in_vocabulary = sum(u)
    print(total_tokens_in_vocabulary)
    return u, total_tokens_in_vocabulary


def calculate_TFIDF(data: pd.DataFrame, vocabulary: list, V_counter: list, total_tokens_in_vocabulary: numpy.array):
    # first calculate the normalized term frequency
    # term frequency is often divided by the total number of terms in the document as a way of normalization
    # in our case instead of documents we have our query vocabulary and our representation vectors

    # basically the frequency of a token in a query is the corresponding value of its representation vector

    for rv in data.representation_vector:
        np.seterr(divide='ignore')
        # normalized term frequency
        total_tokens = np.sum(rv, axis=0)
        tf = np.divide(rv, total_tokens)

        # inverse document frequency
        temp = np.ones(len(vocabulary)) * data.shape[0]
        idf = np.log(np.divide(temp, total_tokens_in_vocabulary))
        # nan values come from the logarithm of 0 so we change them to the value for the rarest tokens 1
        idf = np.nan_to_num(idf, nan=1)

        # TFIDF
        tfidf = tf * idf
        print(tfidf)
        np.seterr(divide='warn')
