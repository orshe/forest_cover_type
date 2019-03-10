import numpy as np
import pandas as pd


def preprocess_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
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
