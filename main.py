import pandas as pd
import csv
import analysis as an
import methods


def read_SQL_share():
    data = pd.read_csv("QueriesWithPlan.csv", delimiter='|', engine='python', error_bad_lines=False,
                       warn_bad_lines=False)
    return data


def readSDSS():
    # used the dollar sign to denote the query statement due to confusion with the quotation marks
    data = pd.read_csv("SDSS_100K_d.csv", delimiter=',', engine='python', encoding='latin1', quotechar="$",
                       error_bad_lines=False)
    return data


if __name__ == '__main__':
    # ------------------------------------------------------
    # read csv files and store errors to da_lines.txt
    csv.field_size_limit(500 * 1024 * 1024)  # because some rows were too big

    print("------------------------ Begining to load the dataset ----------------------------------\n ")

    # data = read_SQL_share()
    data = readSDSS()
    a = data.head(10)

    print("------------------------ dataset loaded-------------------------------------------------\n ")

    V, V_counter = methods.bag_of_ngrams_Vocabulary(5, data)
    u, total_tokens_in_vocabulary = methods.bag_of_ngrams_get_vector(5, data.head(10), V)

    # create a column made out of the representation vectors
    representation_vector = pd.Series(u)
    a.insert(5,"representation_vector", representation_vector.values)

    # print(data.head(10), "\n")
    methods.calculate_TFIDF(a, V, V_counter,total_tokens_in_vocabulary)
    # workload_an = an.Analiser(data)
    # workload_an.analisis(data)
