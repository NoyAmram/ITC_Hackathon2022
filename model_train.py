import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

DATA_FILE = None
MODEL_FILE = None


def read_file(file_name):
    """Receives csv file read it with pandas and return as DataFrame """
    df = pd.read_csv(file_name, encoding="utf-8")
    pass
    return df


def prep(df):
    """Receives DataFrame and preprocess it to be ready for a model
    Returns the data split to X and y"""
    pass


def train(X, y):
    """Receives train data and create model object
    :returns model fitted on train data"""
    pass


def save_model(model, output_file):
    """Receives fitted model and save it to a pickle file"""
    with open(output_file, "wb") as f:
        pickle.dump(model, f)


def main():
    """Starting function to call above functions of the program"""
    df = read_file(DATA_FILE)
    X_train, y_train = prep(df)
    model = train(X_train, y_train)
    save_model(model, MODEL_FILE)


if __name__ == '__main__':
    main()
