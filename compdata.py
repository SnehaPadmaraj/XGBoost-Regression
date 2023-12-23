import os
os.getcwd()
os.chdir(r"C:\Users\sneha\Desktop\machine learning lab")
os.getcwd()
import pandas as pd
import pingouin as pp
dataset = pd.read_csv("Computers.csv")
#find no. of missing values
dataset.isnull().sum()
dataset.head()
pp.anova(dataset, dv = "speed", between = "price")
pp.anova(dataset, dv = "ram", between = "price")
dataset = dataset.drop(["ram", "premium", "multi"], axis = 1)
dataset.head()
import scipy.stats
scipy.stats.pearsonr(dataset["price"], dataset["speed"])
scipy.stats.pearsonr(dataset["price"], dataset["hd"])
scipy.stats.pearsonr(dataset["price"], dataset["screen"])
scipy.stats.pearsonr(dataset["price"], dataset["cd"])
scipy.stats.pearsonr(dataset["price"], dataset["ads"])
scipy.stats.pearsonr(dataset["price"], dataset["trend"])

y = dataset['price']
x = dataset.drop(['price'], axis = 1)
x = pd.get_dummies(x)
pip install xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20)
from xgboost import XGBRegressor
lm = XGBRegressor ()
lm.fit(xtrain,ytrain)
prediction_value = lm.predict(xtest)
from sklearn.metrics import r2_score
r2_score (ytest, prediction_value)
