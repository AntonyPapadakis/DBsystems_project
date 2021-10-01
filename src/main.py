import numpy as np
import pandas as pd
import csv
import analysis as an
import methods
from matplotlib import pyplot as plt
import sys

plt.style.use('dark_background')


def read_SQL_share():
    data = pd.read_csv("QueriesWithPlan.csv", delimiter='|', engine='python', error_bad_lines=False,
                       warn_bad_lines=False)
    return data


def readSDSS():
    # used the dollar sign to denote the query statement due to confusion with the quotation marks
    data = pd.read_csv("SDSS_100K_d_mix.csv", delimiter=',', engine='python', encoding='latin1', quotechar="$",
                       error_bad_lines=False)
    return data


if __name__ == '__main__':

    print(f"Arguments count: {len(sys.argv)}")
    repr_level = ""
    dataset = ""
    dir_load = ""
    count_args = 0
    for arg in sys.argv:
        if arg == "-r":  # argument flag for representation level char or word
            repr_level = sys.argv[count_args + 1].strip().lower()
        elif arg == "-d":  # argument flag for dataset to be used
            dataset = sys.argv[count_args + 1].strip().lower()
        elif arg == "-f":  # argument flag for folder containing vector representation to be loaded
            dir_load = sys.argv[count_args + 1].strip()
        count_args += 1

    # ------------------------------------------------------
    # read csv files and store errors to da_lines.txt
    csv.field_size_limit(500 * 1024 * 1024)  # because some rows were too big

    if dataset == "":
        input_not_ok = True
        user_input = ''
        while input_not_ok:
            user_input = input(
                "Please state the dataset you want to use: \nType sdss or sqlshare for the respective dataset\n")
            user_input = user_input.strip().lower()

            if "sdss" in user_input or "sqlshare" in user_input:
                input_not_ok = False

        dataset = user_input
    print("------------------------ Begining to load the dataset ----------------------------------\n ")

    if "sdss" in dataset:
        data = readSDSS()
    elif "sqlshare" in dataset:
        data = read_SQL_share()
    else:
        print("Please specify the dataset to be used. It must be either sqlshare or sdss\n")
        sys.exit(1)

    print("------------------------ dataset loaded-------------------------------------------------\n ")


    # query workload analysis
    workload_an = an.Analiser(data)
    workload_an.analisis(data, False)
    # finished analysis


    if repr_level == "":
        input_not_ok = True
        user_input = ''
        while input_not_ok:
            user_input = input(
                "Please state if you want word level or character level representations to be used: \nType word or char "
                "for the respective dataset\n")
            user_input = user_input.strip().lower()

            if "word" in user_input or "char" in user_input:
                input_not_ok = False

        repr_level = user_input
        print(user_input)
    if "char" not in repr_level and "word" not in repr_level:
        print("Please specify the representation level to be used. Accepted answers are char and word\n")
        sys.exit(2)

    print("dataset: ", dataset, "\nfile containing the representation TFIDF vectors: ", dir_load,
          "\nRepresentation level", repr_level)


    # simple baselines
    methods.baselines(data, dataset)

    # traditional model
    methods.traditional_model(data, dataset, repr_level, dir_load)

    # LSTM model
    regress = True #use this as True when you want to perform regression operations
    methods.neural_net_methods(data, regress, "lstm", repr_level, dataset)
    regress = False #use this as True when you want to perform regression operations
    methods.neural_net_methods(data, regress, "lstm", repr_level, dataset)

    # CNN model
    regress = True #use this as True when you want to perform regression operations
    methods.neural_net_methods(data, regress, "cnn", repr_level, dataset)
    regress = False #use this as True when you want to perform regression operations
    methods.neural_net_methods(data, regress, "cnn", repr_level, dataset)
