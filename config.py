
data_file = "salaries.csv"
model_file = "salaries_model.pkl"

target = 'salary_in_usd'
target_log = 'salary_in_usd_log'

nominal_features = ['company_location', 'job_title',
                    'employment_type', 'company_size']

ordinal_features = ['experience_level']
experience_level_rank = [['EN', 'MI', 'SE', 'EX']]

numerical_features = ['work_year', 'remote_ratio']

# Pipeline logical steps definition
nominal_transformer = Pipeline(steps=[
    ('onehot_encode', OneHotEncoder(dtype=int, sparse=False, handle_unknown='ignore'))])
ordinal_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder(categories=cfg.experience_level_rank,
                                       handle_unknown='error'))])
