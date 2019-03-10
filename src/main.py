import argparse
import warnings

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import preprocess_data

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)


def get_redundant_pairs(data):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(data, n=5):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


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

    # loading data
    df_train, df_test = preprocess_data()

    data = df_train[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points', 'Cover_Type',
                     'Absolute_Distance_To_Hydrology', 'Rise']].corr()

    # Calculate the correlation coefficients for all combinations
    if args.debug:
        print(data.corr())

    df_correlations = get_top_abs_correlations(data, 12)
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
