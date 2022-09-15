import pickle
import pandas as pd
from flask import Flask, request

MODEL_FILE = None
AWS_PORT = 8080

app = Flask(__name__)

global model


def read_model(file_name):
    """ :param file_name for path to model file
        :returns the fitted model"""
    with open(file_name, "rb") as f:
        global model
        model = pickle.load(f)
    return model


def predict(data):
    """:param data to predict by fitted classification model that was read from pickle file
        :returns model prediction"""
    return model.predict(data)


@app.route('/')
def get_forecast():
    """receives inputs for a single prediction as parameters
    returns a single prediction as a string."""
    # args = request.args.get('arg_name')
    # x = pd.DataFrame([[args]])
    # answer = predict(x)
    # return str(answer[0])
    pass


def main():
    """ starting function to call above functions and verify prediction results"""
    read_model(MODEL_FILE)


if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', port=AWS_PORT, debug=True)