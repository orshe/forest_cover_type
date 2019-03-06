# Scatterplots of aspect, slope and hillshade. This can be helpful to spot structured relationships between input variables.
df_9am = df.drop(columns = ['Elevation', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3',
           'Soil_Type35', 'Soil_Type38', 'Soil_Type39', 'Cover_Type', 'Hillshade_Noon', 'Hillshade_3pm'], axis = 1)
scatter_matrix(df_9am)
plt.show()

df_3pm = df.drop(columns = ['Elevation', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3',
           'Soil_Type35', 'Soil_Type38', 'Soil_Type39', 'Cover_Type', 'Hillshade_Noon', 'Hillshade_9am'], axis = 1)
scatter_matrix(df_3pm)
plt.show()

df_Noon = df.drop(columns = ['Elevation', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3',
           'Soil_Type35', 'Soil_Type38', 'Soil_Type39', 'Cover_Type', 'Hillshade_3pm', 'Hillshade_9am'], axis = 1)
scatter_matrix(df_Noon)
plt.show()
