import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import config as cfg
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def read_file(file_name):
    """Receives csv file read it with pandas and return as DataFrame """
    df = pd.read_csv(file_name, encoding="utf-8")
    # verify no missing values before return
    assert df.isna().sum().mean() == 0
    return df


def log_transformation(df):
    transformer = FunctionTransformer(np.log1p, validate=True)
    transformer.transform(df[[cfg.target]].values)
    return df.drop(columns=[cfg.target], inplace=True)


def split(df):
    """Receives DataFrame and preprocess it to be ready for a model
        Returns the data split to X and y"""
    y = df[cfg.target_log]
    X = df.drop(target_log, axis=1)
    return X, y


def prep(X, y):
    """ Receives X, y and preprocess according to pipeline
    :returns pipe as trained model"""

    preprocess_transformers = ColumnTransformer([
        ('nominal_trans', cfg.nominal_transformer, cfg.nominal_features),
        ('ordinal_trans', cfg.ordinal_transformer, cfg.ordinal_features)
    ])
    f_sel = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=8)
    reg_model = DecisionTreeRegressor()
    pipe = Pipeline(steps=[
        ('preprocess_transformers', preprocess_transformers),
        ('scaler', MinMaxScaler()),
        ('feat_sel', f_sel),
        ('model', reg_model)
    ])
    pipe = pipe.fit(X, y)
    return pipe


def save_model(model, output_file):
    """Receives fitted model and save it to a pickle file"""
    with open(output_file, "wb") as f:
        pickle.dump(model, f)


def main():
    """Starting function to call above functions of the program"""
    salaries = read_file(cfg.data_file)
    salaries = log_transformation(salaries)
    model = prep(split(salaries))
    save_model(model, cfg.model_file)


if __name__ == '__main__':
    main()
