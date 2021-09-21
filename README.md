# ESQL: Efficient Sql queries through exploitation of database leftovers and workloads

This project was developed in the course of the Data Science and Infromation Technologies MSc programme of the NKUA. More specifically during the database systems course. The methodologies in this project where inspired from the paper: Zainab et al. DOI:https://doi.org/10.1145/3318464.3380602

## Datasets

For the experiments conducted in this project we used two large scale query workloads. The first one is comprised of 100000 queries extracted through the casjobs of the SDSS dataset and the second one being the SQLShare released query workload.

## Important note

in the main.py we call all our methods to analyse the workloads and subsequently train our predictive models for query facilitation. However, please note that from the SQLShare dataset we changed the name of the extracted "Query" statement to "statement" and of the "QExecTime" to "busy". Also the Query - statement values coming from the SDSS where enclosed with dolar signs '$' when extracted from casjobs to avoid any string handling problems with quotation marks (default extraction option from SDSS was with "", but there were collisions with "" used in the queries themeshelves).

## Run details

When trying to run main in order to reproduce exactly our experiments please note that the dataset names for SDSS and SQLShare are hardcoded as "SDSS_100K_d.csv" and "QueriesWithPlan.csv" and that main will only accept these two datasets. Feel free to experiment on the code or use any of the method.py functions or the Analiser class in analysis.py.

main.py supports arguments: 

main.py -r [representation level for models char or word] -d [dataset_filepath] -f [file with representation vectors for traditional approach]

all arguments are optional.

If you have already trained your model and have stored your train and test data you may use the evaluation.py script as:

evaluation.py -x [X_test data filename] -y [Y_test data filename] -f [path for trained model] -t [OPTIONAL - input any string and a baseline model will be calculated]

Evaluation.py calculates a number of metrics for each of the problems, classification or regression.

