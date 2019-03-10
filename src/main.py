import argparse
import warnings

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import preprocess_data, get_top_abs_correlations

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)


def evaluate_models(models, X, Y, scoring='accuracy'):
    results = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=12345)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append((name, cv_results))

    return results


if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    df_train, df_test = preprocess_data()

    df_correlations = get_top_abs_correlations(df_train)
    if args.debug:
        print(df_correlations)

    # Compare the models -> which one gives the best accuracy with our data
    models = [('DT', DecisionTreeClassifier()),
              ('RFC', RandomForestClassifier()),
              ('LR', LogisticRegression()),
              ('ETC', ExtraTreesClassifier())]

    X = df_train.drop(['Cover_Type'], axis=1)
    Y = df_train['Cover_Type']

    # evaluate each model in turn
    results = evaluate_models(models, X, Y)

    for name, results in results:
        print(f"{name}: {results.mean()}({results.std()})")

    # DT: 0.716071 (0.049579)
    # RFC: 0.765212 (0.046407)
    # LR: 0.621958 (0.065839)
    # ETC: 0.765542 (0.044487)
