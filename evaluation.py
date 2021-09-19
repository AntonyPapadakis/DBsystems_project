import sys
import numpy as np
from numpy import save, load
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, explained_variance_score, \
    mean_squared_error, max_error

if __name__ == '__main__':
    """
    this script takes as input the trained model and the corresponding test set both values and labels (X , Y)
    and provides evaluation metrics e.g. accuracy, MSE and the corresponding graphs
    """

    print(f"Arguments count: {len(sys.argv)}")
    if len(sys.argv) < 7:
        print("not all arguments were given please try again\n")
        sys.exit(1)

    model_path = ""  # the given model should contain the full run path
    X_test = ""  # test data
    Y_test = ""  # test data labels
    model_type = ""  # type string
    count_args = 0
    for arg in sys.argv:
        if arg == "-x":  # argument flag for test data
            X_test = sys.argv[count_args + 1].strip().lower()
        if arg == "-y":  # argument flag for test data labels
            Y_test = sys.argv[count_args + 1].strip().lower()
        elif arg == "-f":  # argument flag for model path
            model_path = sys.argv[count_args + 1].strip()
        elif arg == "-t":  # argument flag for type of model given e.g "word traditional sdss cpu"  or "char lstm
            # sdss cpu" ...
            model_type = sys.argv[count_args + 1].strip()
        count_args += 1

    if "word" in model_path:
        repr_level = "word"
    else:
        repr_level = "char"

    if X_test != "":
        pathX= X_test
        X_test = load(X_test)  # load data
    if Y_test != "":
        pathY= Y_test
        Y_test = load(Y_test)  # load labels


    # now time for the evaluation protocol
    print(Y_test)

    if "Log" in model_path or "huber" in model_path:

        model_type = repr_level + " traditional model "
        print(model_path)
        model = joblib.load(model_path)
        if "error" in model_path and "cnn" in pathX or "lstm" in pathX :
            Y_test = np.argmax(Y_test, axis=1)
            X_test = X_test.reshape((len(X_test),len(X_test[0])))

        Y_pred = model.predict(X_test)

    else:
        from tensorflow import keras

        model_type = repr_level + " neural net model "

        model = keras.models.load_model(model_path)
        Y_pred = model.predict(X_test)
        if "error" in model_path:
            Y_pred = np.argmax(Y_pred, axis=1)
            Y_test = np.argmax(Y_test, axis=1)

    print(Y_pred)
    print(Y_test)

    count1 = 0
    count11 = 0

    for i in range(0, len(Y_test)):
        if Y_test[i] == -1:
            count1 += 1
        elif Y_test[i] == 1:
            count11 += 1
    print(count1)
    print(count11)

    # classification evaluation
    if "error" in model_path:

        if "sdss" in model_path:
            model_type = model_type + " sdss error classification "
        else:
            model_type = model_type + " sqlshare error classification"

        print("Given model type is:", model_type, "\nmodel name: ", model_path)

        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, labels=[0, 1, -1], average=None)
        pr = precision_score(Y_test, Y_pred, average=None)
        rc = recall_score(Y_test, Y_pred, labels=[0, 1, -1], average=None)

        #   count missed values
        missed = len([i for i in range(0, len(Y_test)) if Y_test[i] != Y_pred[i]])
        print(model_type, "- Accuracy score: ", acc)
        print(model_type, "- F1 score: ", f1)
        print(model_type, "- Precision score: ", pr)
        print(model_type, "- Recall score: ", rc)
        print(model_type, "- Missed values: ", missed)

    else:

        if "sdss" in model_path:
            model_type = model_type + " sdss regression"
        else:
            model_type = model_type + " sqlshare regression"

        print("Given model type is:", model_type, "\nmodel name: ", model_path)

        ev = explained_variance_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        me = max_error(Y_test, Y_pred)

        print(model_type, "- Mean Squared Error score: ", mse)
        print(model_type, "- Explained Variance score: ", ev)
        print(model_type, "- Max Error score: ", me)
