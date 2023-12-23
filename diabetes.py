import os
os.getcwd()
os.chdir(r"C:\Users\sneha\Desktop\machine learning lab")
os.getcwd()
import pandas as pd
data = pd.read_csv("diabetes.csv")
data.head()
data.isnull().sum()
import pingouin as pp
pp.anova(data, dv="Outcome", between = "Pregnancies")
pp.anova(data, dv="Outcome", between = "Glucose")
pp.anova(data, dv="Outcome", between = "BloodPressure")
pp.anova(data, dv="Outcome", between = "Age")
pp.anova(data, dv="Outcome", between = "Insulin")
pp.anova(data, dv="Outcome", between = "BMI")
pp.anova(data, dv="Outcome", between = "DiabetesPedigreeFunction")
pp.anova(data, dv="Outcome", between = "SkinThickness")
data.drop(['SkinThickness'], axis = 1)
y = data["Outcome"]
x = data.drop(["Outcome"], axis = 1)
x = pd.get_dummies(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBRegressor
xtrain,ytrain,xtest,ytest = train_test_split(x,y, test_size = 0.20)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(xtrain, ytrain)
pred_val = xgb_model.predict(xtest)
accuracy = accuracy_score(ytest,pred_val)
accuracy
conf_matrix = confusion_matrix(ytest, pred_val)
conf_matrix
