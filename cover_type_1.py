import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

# https://www.kaggle.com/c/forest-cover-type-kernels-only

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
df_train = df_train.drop(['Id'], axis=1)

df_train['Absolute_Distance_To_Hydrology'] = (df_train['Horizontal_Distance_To_Hydrology'] ** 2 + df_train[
    'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
df_train['Absolute_Distance_To_Hydrology'] = df_train['Absolute_Distance_To_Hydrology'].astype(int)
df_train['Rise'] = df_train['Elevation'] * np.sin(df_train['Slope'] * np.pi / 180.0)
df_train['Rise'] = df_train['Rise'].astype(int)

print(df_train.sample(5))

# datatypes of the attributes
print(df_train.dtypes)

# we need to see all the columns
pd.set_option('display.max_columns', None)

# Descriptive statistics for each column
print(df_train.describe(include='all'))

# Removing Soil_type 7 and 15 -> each value is zero
df_train = df_train.drop(['Soil_Type7', 'Soil_Type15'], axis=1)
df_test = df_test.drop(['Soil_Type7', 'Soil_Type15'], axis=1)

# Correlation matrix (heatmap) - Correlation requires continuous data -> ignore Wilderness_Area and Soil_Type as they are binary values

corrmat = df_train.drop(['Wilderness_Area1',
                         'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                         'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
                         'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
                         'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
                         'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
                         'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
                         'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                         'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                         'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
                         'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
                         'Soil_Type39', 'Soil_Type40'], axis=1).corr()

# f, ax = plt.subplots()
# sns.heatmap(corrmat, vmax=.8, square=True, cmap="Greens")
# plt.show()

# Correlation values
data = corrmat

# Calculate the correlation coefficients for all combinations
print(data.corr())


def get_redundant_pairs(data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
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


df_correlations = get_top_abs_correlations(data, 12)
print(df_correlations)

# for x, y in df_correlations.axes[0].tolist():
#     sns.pairplot(data=df_train, hue='Cover_Type', x_vars=x, y_vars=y, palette="Set2")
#     plt.show()
#
#     sns.lmplot(x=x, y=y, hue='Cover_Type', data=df_train, markers='o', palette="Set2", lowess=True)
#     plt.show()

# Compare the models -> which one gives the best accuracy with our data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection

# prepare models
models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('SVM', SVC()))
models.append(('GB', GaussianNB()))

X = df_train.drop(['Cover_Type'], axis=1)
Y = df_train['Cover_Type']

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=12345)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# DT: 0.716071 (0.049579)
# RFC: 0.765212 (0.046407)
# LR: 0.621958 (0.065839)
# ETC: 0.765542 (0.044487)
# SVM: 0.020172 (0.023771)
# GB: 0.568783 (0.094286)