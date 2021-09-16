import pandas as pd
import re
from nltk.util import ngrams
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, Conv1D, MaxPooling1D, \
    Flatten
from sklearn.linear_model import HuberRegressor
from sklearn import preprocessing
from numpy import save, load
import sys
import joblib
import time


def map_to_vector(V, s):
    """
    :param V: vocabulary
    :param s: list of ngrams for a certain query
    :return: the representation vector which is a sum of all of the onehot encoded vectors of each ngram in s
    with respect to the vocabulary V
    """

    representation_vector = np.zeros(len(V))
    VL = len(V)

    """
    indices_temp = [([index for index, val in enumerate(V) if val == tok] if tok in V else [None]) for tok in s]
    indices = [index[0] for index in indices_temp if index[0] is not None]

    onehot_vector = np.zeros(VL)
    np.add.at(onehot_vector, indices, 1)
    representation_vector = np.add(representation_vector, onehot_vector)
    """

    for t in s:
        if t in V:
            onehot_vector = np.zeros(VL)
            onehot_vector[V.index(t)] = 1
            representation_vector = np.add(representation_vector, onehot_vector)

    return representation_vector


def ngrams_from_query(n: int, q: str, result: list, repr_level: str):
    """

    :param result: is a list containing all the different ngrams from all queries
    :output: is a list containing ngrams only for the given query

    """
    q = q.lower()
    if "word" in repr_level:
        # regular expression to replace all digits and whitespace chars with <d> token but not alphanumerics
        q = re.sub(r'\b[0-9]+\b|\s', ' <d> ', q)
        tokens = [wtoken for wtoken in q.split(" ") if wtoken != ""]
    elif "char" in repr_level:
        tokens = [ctoken for ctoken in q if ctoken != ""]

    output = []
    for num_of_ngrams in range(2, n + 1):  # we find the most frequent ngrams up to the given n
        # getting the ngrams and storing them
        output = list(ngrams(tokens, num_of_ngrams))  # bigrams, trigrams ...
        result.extend(output)
    return result, output


def bag_of_ngrams_Vocabulary(n: int, data: pd.DataFrame, repr_level: str) -> (list, list, list):
    result = []
    keep_ngrams_per_q = []
    for q in data.statement:
        result, ngrams_from_each_q = ngrams_from_query(n, q, result, repr_level)
        keep_ngrams_per_q.append(ngrams_from_each_q)

    out = Counter(result)
    count_items = 0
    Vocabulary = []
    V_counter = []

    if "word" in repr_level:
        tok_freq_floor = 300
    else:
        tok_freq_floor = 1700  # we increase this number for char level to limit a bit the vocabulary

    for key, value in out.items():  # iterating over  the dictionaries' items

        if value >= tok_freq_floor:  # appending the most frequent ngrams to the vocabulary
            Vocabulary.append(key)
            V_counter.append(value)
            count_items += 1

    print("items counted in the Vocabulary", count_items)
    print("The Vocabulary: ", Vocabulary)

    return Vocabulary, V_counter, keep_ngrams_per_q


def bag_of_ngrams_get_vector(vocabulary: list, ngrams_per_query: list) -> (list, list):
    # for every query calculate the ngrams and their one hot encoding with
    # respect to each ngram in the Vocabulary
    # and then add them to create the vector representation for each query

    # declaring list of representation vectors
    # queries are mapped to u dimmensional vectors
    u = []
    for i in range(0, len(ngrams_per_query)):
        u.append(map_to_vector(vocabulary, ngrams_per_query[i]))

    total_tokens_in_vocabulary = sum(u)
    print("\ntotal tokens in V: ", total_tokens_in_vocabulary, "\n", flush=True)
    return u, total_tokens_in_vocabulary


def calculate_TFIDF(data: pd.DataFrame, vocabulary: list, total_tokens_in_vocabulary: np.array) -> np.array:
    # first calculate the normalized term frequency
    # term frequency is often divided by the total number of terms in the document as a way of normalization
    # in our case instead of documents we have our query vocabulary and our representation vectors

    # basically the frequency of a token in a query is the corresponding value of its representation vector
    rep_vectors = np.zeros((data.shape[0], len(vocabulary)))
    counter = 0  # counter used to insert the tfidf representation vector to the object to be returned
    for rv in data.representation_vector:
        np.seterr(divide='ignore', invalid='ignore')
        # normalized term frequency
        total_tokens = np.sum(rv, axis=0)
        tf = np.divide(rv, total_tokens)
        # replace NaN values
        tf = np.nan_to_num(tf, nan=0)

        # inverse document frequency
        temp = np.ones(len(vocabulary)) * data.shape[0]
        idf = np.log(np.divide(temp, total_tokens_in_vocabulary))
        # nan values come from the logarithm of 0 so we change them to the value for the rarest tokens 1
        idf = np.nan_to_num(idf, nan=1)

        # TFIDF
        tfidf = tf * idf

        np.seterr(divide='warn', invalid='warn')

        # insert rep vector to the numpy array
        rep_vectors[counter][:] = tfidf[:]
        counter += 1

    return rep_vectors


def baselines(data: pd.DataFrame):
    # simple baselines
    # most frequent value of label predicted for classification problems
    # error label problem
    unique_elements, counts_elements = np.unique(data.error.values, return_counts=True)
    mfreq_index = np.where(counts_elements.max() == counts_elements)  # most frequent
    mfreq_error = unique_elements[mfreq_index][0]
    print("most frequent error label: ", mfreq_error, "\n")

    # median value for regression problems
    # CPU time busy problem
    median_cpu = np.median(data.busy.values)
    print("median cpu time: ", median_cpu, "\n")

    # median value for regression problems
    # Answer size problem
    median_ans_size = np.median(data.rows.values)
    print("median answer size: ", median_ans_size, "\n")


def logistic_regression_classification(X_train, X_test, Y_train, Y_test, repr_level):
    # changed max_iter due to error
    error_classifier = LogisticRegression(multi_class='ovr', random_state=0, max_iter=200)
    scores = cross_val_score(error_classifier, X_train, Y_train, cv=5)
    print("\ncrossval scores", scores)
    error_classifier.fit(X_train, Y_train)
    print("\n", error_classifier.score(X_test, Y_test))

    filename = repr_level + '_sdss_LogReg_error_classifier.sav'
    joblib.dump(error_classifier, filename)


def traditional_model(data: pd.DataFrame, dataset: str, repr_level: str, dir_load: str):
    if dir_load == "":
        input_not_ok = True
        user_input = ''
        while input_not_ok:
            user_input = input(
                "Please state if you want the representation vectors to be loaded from disk (provided there is a .npy "
                "data file saved in the current directory)\nType yes or no for the respective answer\n")
            user_input = user_input.strip().lower()

        if user_input == "yes" or user_input == "no":
            input_not_ok = False

        answer = user_input
        dir_load = ""
    else:
        answer = "yes"
        print("\nYou have provided a file containing the TFIDF vector representations to be used for training and "
              "testing.\nPlease note that if the file is invalid an error will occur.\n ")

    if "no" in answer:

        V, V_counter, ngrams_per_query = bag_of_ngrams_Vocabulary(5, data, repr_level)
        u, total_tokens_in_vocabulary = bag_of_ngrams_get_vector(V, ngrams_per_query)

        # create a column made out of the representation vectors
        representation_vector = pd.Series(u)
        data.insert(5, "representation_vector", representation_vector.values)

        # get the training feature vectors
        X = calculate_TFIDF(data, V, total_tokens_in_vocabulary)

        # save representation vectors TFIDF to npy file
        save(repr_level + '_representationTFIDF_vectors_' + dataset + '.npy', X)
    elif "yes" in answer:
        if dir_load == "":
            X = load(repr_level + '_representationTFIDF_vectors_' + dataset + '.npy')
        else:
            X = load(dir_load)
    else:
        sys.exit(3)

    # ---------------------------------------------------------------------------------------------------
    # error
    # ---------------------------------------------------------------------------------------------------

    # get the error labels
    Y_error = data.error.values

    # classification split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_error, test_size=0.1, random_state=5)

    # train model for the error prediction - classification problem
    start = time.time()
    logistic_regression_classification(X_train, X_test, Y_train, Y_test, repr_level)
    end = time.time()
    print("\nLogistic regression time is: ", end - start)

    # Regression
    # normalize the answer size and CPU time labels (regression labels)
    # normalization equation  y′i = ln(yi + ϵ − min(y))
    epsilon = 1  # a constant epsilon set to 1 to prevent the input of the ln function to be 0

    # ---------------------------------------------------------------------------------------------------
    # CPU time
    # ---------------------------------------------------------------------------------------------------

    b = epsilon - data.busy.values.min()
    CPU_time = data.busy.values
    np.add.at(CPU_time, [*range(0, len(CPU_time))], b)
    CPU_time = np.divide(CPU_time, data.busy.values.max() - data.busy.values.min())
    # replace NaN values
    CPU_time = np.nan_to_num(CPU_time, nan=0)
    Y_cpu = np.log(CPU_time)

    # cpu time regression split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_cpu, test_size=0.1, random_state=6)
    huber_cpu = HuberRegressor(max_iter=200, tol=1e-3)
    scores = cross_val_score(huber_cpu, X_train, Y_train, cv=5)
    print("\nhuber cpu crossval scores", scores)

    start = time.time()
    huber_cpu.fit(X_train, Y_train)
    end = time.time()
    print("\nHuber regression for cpu time model training time is: ", end - start)

    print("\nhuber cpu", huber_cpu.score(X_test, Y_test))
    print(huber_cpu.predict(X[:5]), " true: ", Y_cpu[:5], "\n")
    filename = repr_level + '_sdss_huber_regressor_cpu.sav'
    joblib.dump(huber_cpu, filename)

    # ---------------------------------------------------------------------------------------------------
    # answer size
    # ---------------------------------------------------------------------------------------------------

    b = epsilon - data.rows.values.min()
    Ans_size = data.rows.values
    np.add.at(Ans_size, [*range(0, len(Ans_size))], b)
    Ans_size = np.divide(Ans_size, data.rows.values.max() - data.rows.values.min())
    # replace NaN values
    Ans_size = np.nan_to_num(Ans_size, nan=0)
    Y_answer = np.log(Ans_size)

    # answer size regression split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_answer, test_size=0.1, random_state=7)
    huber_ans_size = HuberRegressor(max_iter=200, tol=1e-3)
    scores = cross_val_score(huber_ans_size, X_train, Y_train, cv=5)
    print("\nhuber answer size crossval scores", scores)

    start = time.time()
    huber_ans_size.fit(X_train, Y_train)
    end = time.time()
    print("\nHuber regression for answer size model training time is: ", end - start)

    print("\nhuber answer size", huber_ans_size.score(X_test, Y_test))
    print(huber_ans_size.predict(X[:5]), " true: ", Y_answer[:5])
    filename = repr_level + '_sdss_huber_regressor_answer_size.sav'
    joblib.dump(huber_ans_size, filename)


def preprocess_for_neural_net_models(data: pd.DataFrame, X_train: np.array, token_level: str) -> (np.array, np.array):
    Y_error = data.error.values

    max_words = 1500
    if "word" in token_level:
        for i in range(0, X_train.shape[0]):
            X_train[i] = X_train[i].lower()
            # regular expression to replace all digits and whitespace chars with <d> token but not alphanumerics
            X_train[i] = re.sub(r'\b[0-9]+\b|\s', ' <d> ', X_train[i])

        phrase_len = data.statement.apply(lambda p: len(p.split(' ')))
    else:
        for i in range(0, X_train.shape[0]):
            X_train[i] = X_train[i].lower()
        phrase_len = data.statement.apply(lambda p: len(p))

    max_phrase_len = phrase_len.max()

    tokenizer = Tokenizer(num_words=max_words, )

    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_phrase_len)
    Y_train = to_categorical(Y_error)

    print(X_train.shape, " ", Y_train.shape)
    print(X_train, "\n", Y_train)

    # accordingly shape the input shape
    X_train = np.asarray(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    return X_train, Y_train


def neural_net_methods(data: pd.DataFrame, regress: bool, net_type: str, token_level: str):
    # clean values
    X_train = data.statement.values
    print(X_train.shape[0])
    X_train, Y_train = preprocess_for_neural_net_models(data, X_train, token_level)

    input_sh = (X_train.shape[1], 1)

    batch_size = 32
    epochs = 2

    model = Sequential()

    if "lstm" in net_type:
        X_train = X_train.astype(np.float)

        # model_lstm.add(Embedding(input_dim=max_words, input_length=X_train.shape[1], output_dim=32))

        model.add(LSTM(32, return_sequences=True, input_shape=input_sh))
        model.add(LSTM(16, return_sequences=True, ))
        model.add(LSTM(8, dropout=0.3))
    else:
        model_cnn = Sequential()

        # model_cnn.add(Embedding(input_dim=max_words, input_length=X_train.shape[1], output_dim=32))

        model.add(Conv1D(input_shape=input_sh, filters=32, kernel_size=10, activation='relu'))
        model.add(MaxPooling1D(pool_size=5))
        # need the flatten layer to move the data on to a fully connected layer
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))

    if regress:
        model.add(Dense(1, activation='linear'))
    else:
        model.add(Dense(2, activation='sigmoid'))

    if regress:

        epsilon = 1

        for case in range(0, 2):

            if case == 0:
                # ---------------------------------------------------------------------------------------------------
                # CPU time
                # ---------------------------------------------------------------------------------------------------

                b = epsilon - data.busy.values.min()
                CPU_time = data.busy.values
                np.add.at(CPU_time, [*range(0, len(CPU_time))], b)
                CPU_time = np.divide(CPU_time, data.busy.values.max() - data.busy.values.min())
                # replace NaN values
                CPU_time = np.nan_to_num(CPU_time, nan=0)
                Y_cpu = np.log(CPU_time)
                Y_train = Y_cpu

            else:
                # ---------------------------------------------------------------------------------------------------
                # answer size
                # ---------------------------------------------------------------------------------------------------

                b = epsilon - data.rows.values.min()
                Ans_size = data.rows.values
                np.add.at(Ans_size, [*range(0, len(Ans_size))], b)
                Ans_size = np.divide(Ans_size, data.rows.values.max() - data.rows.values.min())
                # replace NaN values
                Ans_size = np.nan_to_num(Ans_size, nan=0)
                Y_answer = np.log(Ans_size)
                Y_train = Y_answer

            model.compile(loss='huber_loss', optimizer='Adam', metrics=["mean_squared_error"])
            start = time.time()
            history = model.fit(X_train, Y_train, validation_split=0.1, epochs=2, batch_size=batch_size)
            end = time.time()

            if case == 0:
                print("\n", token_level, " ", net_type, " Regression for cpu time model", "time is: ", end - start)
            else:
                print("\n", token_level, " ", net_type, " Regression for answer size model", "time is: ", end - start)
    else:
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        start = time.time()
        history = model.fit(X_train, Y_train, validation_split=0.1, epochs=2, batch_size=batch_size)
        end = time.time()
        print("\n", token_level, " ", net_type, " Classification for query error prediction", "time is: ", end - start)
