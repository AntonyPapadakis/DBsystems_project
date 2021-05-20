from io import StringIO
import pandas as pd
import sys
import csv


def read_SQL_share():
    data = pd.read_csv("QueriesWithPlan.csv", delimiter='|', engine='python', error_bad_lines=False)
    return data


def readSDSS():
    data = pd.read_csv("SDSS_queries_80K.csv", delimiter=',', engine='python', error_bad_lines=False)
    return data


if __name__ == '__main__':

    #------------------------------------------------------
    #read csv files and store errors to da_lines.txt
    csv.field_size_limit(500 * 1024 * 1024)  # because some rows were too big

    old_stderr = sys.stderr  # standard stream keep
    result = StringIO()  # new stream
    sys.stderr = result  # use this as err stream for read csv

    #data = read_SQL_share()
    data = readSDSS()
    print(data.head(100).statement)

    sys.stderr = old_stderr  # reset err
    result_string = result.getvalue()  # write errors to bad_lines.txt
    with open('bad_lines.txt', 'w') as bad_lines:
        for line in result_string.split(r'\n'):
            bad_lines.write(line)
            bad_lines.write('\n')

    #-------------------------------------------------------
