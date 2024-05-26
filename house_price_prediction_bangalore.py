import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os


# fetching the csv file to a pandas dataframe
Bangalore_dataset = pd.read_csv('Bangalore.csv')

# Dropping the columns
Bangalore_dataset.drop(["MaintenanceStaff","Gymnasium","LandscapedGardens","RainWaterHarvesting","IndoorGames",
                        "Intercom","PowerBackup","Cafeteria","MultipurposeRoom","Wifi","Children'splayarea","LiftAvailable",
                        "VaastuCompliant","GolfCourse","Wardrobe","ShoppingMall","ATM","School","24X7Security"],axis=1,inplace=True)

# loading the label encoder for location
le = joblib.load('models/Bangalore_label_encoder.joblib')

# finally Loading the models to predict the House Price
models = []
models_name = ['Linear Regression','K-Nearest Neighbors','Decision Tree Regressor','Random Forest Regressor','XG-Boost']

for i in range(5):
    joblibname = "Bangalore_" + models_name[i] + ".joblib"
    location = "models/"+joblibname
    model = joblib.load(location)
    models.append(model)
# print(models)
