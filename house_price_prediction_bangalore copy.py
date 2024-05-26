# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os


# fetching the csv file to a pandas dataframe
Bangalore_dataset = pd.read_csv('Bangalore.csv')

# checking the dataset shape, parameters
Bangalore_dataset.shape

# Bangalore_dataset.info()

# other than price,Area,No.of bedrooms
# 0 - Absent, 1 - Present, 9 - Not mentioned
Bangalore_dataset.describe()

"""After checking the description of the dataset, it can be said that majority of the parameters don't 
have appropraite values so including them may bring about misleading results, so  in this case we won't be taking 
that into considerations"""

Bangalore_dataset.head()

# observing the Price vs Area plot, this will help in determinine non important data
Price = np.log(Bangalore_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Bangalore_dataset, fit_reg = False)

Bangalore_dataset.drop(Bangalore_dataset[Bangalore_dataset['Area']>=4000].index,inplace = True)
Price = np.log(Bangalore_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Bangalore_dataset, fit_reg = False)

sns.countplot(x = Bangalore_dataset['JoggingTrack'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['JoggingTrack'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['Gymnasium'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['IndoorGames'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['SwimmingPool'],data = Bangalore_dataset)

"""From the rest of the dataset also it can be said that a  lot of data is missing so we may need to delete a majority of data"""

Bangalore_dataset.replace(9,np.nan,inplace=True)
Bangalore_dataset.dropna(axis = 0,how="any",inplace=True)

"""this is how we can get the correlation between various parameters."""

# counting duplicates and removing them later
Bangalore_dataset.duplicated().sum()

# getting rid of the duplicates and verify
Bangalore_dataset.drop_duplicates(inplace=True)
Bangalore_dataset.shape

Bangalore_dataset.drop(["MaintenanceStaff","Gymnasium","LandscapedGardens","RainWaterHarvesting","IndoorGames",
                        "Intercom","PowerBackup","Cafeteria","MultipurposeRoom","Wifi","Children'splayarea","LiftAvailable",
                        "VaastuCompliant","GolfCourse","Wardrobe","ShoppingMall","ATM","School","24X7Security"],axis=1,inplace=True)

Bangalore_dataset

Bangalore_dataset.describe()

# label encoding the locations before moving on to the splitting
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# Bangalore_dataset["Location"] = le.fit_transform(Bangalore_dataset["Location"])
le = joblib.load('models/Bangalore_label_encoder.joblib')

Bangalore_dataset.describe()

# now the dependent variables and independent variables matrices
# X = Bangalore_dataset.iloc[:,1:].values
# y = Bangalore_dataset.iloc[:,0].values
# y = y/100000   # converting prices into lakhs
# print(X)
# print(y)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

models = []
models_name = ['Linear Regression','K-Nearest Neighbors','Decision Tree Regressor','Random Forest Regressor','XG-Boost']

for i in range(5):
    joblibname = "Bangalore_" + models_name[i] + ".joblib"
    location = "models/"+joblibname
    model = joblib.load(location)
    models.append(model)
print(models)
models_directory = 'models/'
