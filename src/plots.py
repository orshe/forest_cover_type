
# Correlation matrix (heatmap) - Correlation requires continuous data -> ignore Wilderness_Area and Soil_Type as they
# are binary values def plot_correlations(df_correlations, df_train): f, ax = plt.subplots() sns.heatmap(corrmat,
# vmax=.8, square=True, cmap="Greens") plt.show()
#
#     for x, y in df_correlations.axes[0].tolist():
#         sns.pairplot(data=df_train, hue='Cover_Type', x_vars=x, y_vars=y, palette="Set2")
#         plt.show()
#
#         sns.lmplot(x=x, y=y, hue='Cover_Type', data=df_train, markers='o', palette="Set2", lowess=True)
#         plt.show()
