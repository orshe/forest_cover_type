import numpy as np
import pandas as pd


def preprocess_data():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_train = df_train.drop(['Id'], axis=1)

    df_train['Absolute_Distance_To_Hydrology'] = (df_train['Horizontal_Distance_To_Hydrology'] ** 2 + df_train[
        'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    df_train['Absolute_Distance_To_Hydrology'] = df_train['Absolute_Distance_To_Hydrology'].astype(int)
    df_train['Rise'] = df_train['Elevation'] * np.sin(df_train['Slope'] * np.pi / 180.0)
    df_train['Rise'] = df_train['Rise'].astype(int)

    # Removing Soil_type 7 and 15 -> each value is zero
    df_train = df_train.drop(['Soil_Type7', 'Soil_Type15'], axis=1)
    df_test = df_test.drop(['Soil_Type7', 'Soil_Type15'], axis=1)

    return df_train, df_test


def get_redundant_pairs(data):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(data, n=12):
    data = data[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                 'Horizontal_Distance_To_Fire_Points', 'Cover_Type',
                 'Absolute_Distance_To_Hydrology', 'Rise']].corr()

    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
