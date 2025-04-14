import xgboost as xgb


def train_model(data, params):

    task = params['task']
    est = params['model']['n_estimators']
    max_depth = params['model']['max_depth']
    random_state = params['model']['random_state']

    if task == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=est, max_depth=max_depth, random_state=random_state)
        X = data.iloc[:, :-2]
        y = data.iloc[:, -2]
        from IPython import embed; embed()
        model.fit(X, y)

    return model

        