from utils.utils import create_directory
from utils.utils import read_dataset
import os
import numpy as np
import sklearn
import utils
from classifiers import fcn

def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier = fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=False)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)

def predict():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier = fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose=False)

    df_metrics=classifier.predict(x_test,y_true, x_train, y_train, y_test, return_df_metrics=True)


############################################### main
root_dir = '../data'
archive_name = 'shapelet'
for dataset_name in utils.constants.UCR12:
        output_directory = root_dir + '/results/' + dataset_name + '/'
        test_dir_df_metrics = output_directory + 'df_metrics.csv'

        print(archive_name, dataset_name)
        if os.path.exists(test_dir_df_metrics):
            print('model exist. Using the exist model')
            datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
            predict()
        else:
            create_directory(output_directory)
            datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
            fit_classifier()


