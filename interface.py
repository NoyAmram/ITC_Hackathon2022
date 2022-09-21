import pickle
import pandas as pd
from flask import Flask, request
import config as cfg
import numpy as np

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
    return np.expm1(model.predict(data))


def refresh():
    """Upload new model file every time it is updated and trained on new data"""
    global model
    model = pickle.load(cfg.model_file)


@app.route('/')
def get_forecast():
    """receives inputs for a single prediction as parameters
    returns a single prediction as a string."""
    work_year = int(request.args.get('work_year'))
    experience_level = request.args.get('experience_level')
    employment_type = request.args.get('employment_type')
    job_title = request.args.get('job_title')
    remote_ratio = int(request.args.get('remote_ratio'))
    company_location = request.args.get('company_location')
    company_size = request.args.get('company_size')

    x = pd.DataFrame(data=
                     {'work_year': work_year,
                      'experience_level': experience_level,
                      'employment_type': employment_type,
                      'job_title': job_title,
                      'remote_ratio': remote_ratio,
                      'company_location': company_location,
                      'company_size': company_size
                      }, index=[0])
    answer = np.round(predict(x)/12, 0)
    return str(int(answer[0]))


def main():
    """ starting function to call above functions and print prediction results"""
    read_model(cfg.model_file)


if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', port=cfg.AWS_PORT, debug=True)
