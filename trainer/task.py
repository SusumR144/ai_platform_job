import pandas as pd
import numpy as np
import pickle

from google.cloud import storage
BUCKET_NAME = 'intern_susum'
client = storage.Client()

#saving model as pkl file
def local_write_trained_model(model,model_name=" "):
    file =open(str(model_name)+'.pkl', 'wb')
    pickle.dump(model, file)
    file.close()
    print('SAVED THE MODEL SUCESSFULLY\n\n')


#Reading dataset
FILE_PATH = 'gs://intern_susum/CarPrice_Assignment.csv'
df = pd.read_csv(FILE_PATH)


# Data Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns=['CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']

for column in columns:
    df[column]= le.fit_transform(df[column])
print(df.head())

# Data cleaning
print(df.isnull().sum())

# Train Test split
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

y= df.price
X= df.drop('price',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)

#RANDOM FOREST REGRESSION

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
random_forest = RandomForestRegressor(n_estimators=20, random_state=0)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
print(f'r2 score: {np.round(r2_score(y_test, y_pred), 4)*100}%')
local_write_trained_model(random_forest,'trainer/models/random_forest')

random_forest_result = pd.DataFrame({'actual_price':y_test,'predicted_price':y_pred})
random_forest_result.to_csv("RandomForest.csv",index = False)
print(random_forest_result.to_string(index = False))

#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
LR=LinearRegression()
LR.fit(X_train,y_train)
y_pred_1=LR.predict(X_test)

from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE

print('Mean Squared Error:',round(metrics.mean_squared_error(y_test, y_pred_1),2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_1)),2))
print(f'r2 score: {np.round(r2_score(y_test, y_pred_1), 4)*100}%')
local_write_trained_model(LR,'trainer/models/LR')

linear_regression_result = pd.DataFrame({'actual_price':y_test,'predicted_price':y_pred_1})
linear_regression_result.to_csv("LinearRegression.csv",index = False)
print(linear_regression_result.to_string(index = False))

#XGBOOST REGRESSION

import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4,
                max_depth = 24, alpha = 5, n_estimators = 200)
xg_reg.fit(X_train,y_train)
y_pred_2 = xg_reg.predict(X_test)

from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE

print('Mean Squared Error:',round(metrics.mean_squared_error(y_test, y_pred_2),2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_2)),2))
print(f'r2 score: {np.round(r2_score(y_test, y_pred_2), 4)*100}%')
local_write_trained_model(xg_reg,'trainer/models/xg_reg')

xg_result = pd.DataFrame({'actual_price':y_test,'predicted_price':y_pred_2})
xg_result.to_csv("xgboost.csv",index = False)
print(xg_result.to_string(index = False))


#HYPERPARAMETER TUNING FOR BEST MODEL 

from sklearn.model_selection import GridSearchCV
forest = RandomForestRegressor(n_jobs=-1)
forest.fit(X_train, y_train)
param_grid = [
{'n_estimators': [10, 25], 'max_features': [5, 10], 
 'max_depth': [10, 50, None], 'bootstrap': [True, False]}
]
print(param_grid)

grid_search_forest = GridSearchCV(forest, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_forest.fit(X_train, y_train)

cvres = grid_search_forest.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
print(grid_search_forest.best_estimator_)
print(grid_search_forest.best_params_)

# Performance metrics
grid_best= grid_search_forest.best_estimator_.predict(X_train)
errors = abs(grid_best - y_train)
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_train))
# Calculate and display accuracy
accuracy = 100 - mape    
#print result
print('The best model from grid-search has an accuracy of', round(accuracy, 2),'%')

# RMSE value
from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_train, grid_best)
final_rmse = np.sqrt(final_mse)
print('The best model from the grid search has a RMSE of', round(final_rmse, 2))

#FEATURE IMPORTANCE

# extract the numerical values of feature importance from the search
importances = grid_search_forest.best_estimator_.feature_importances_

#create a feature list from the original dataset (list of columns)

feature_list = list(X.columns)

#create a list of tuples
feature_importance= sorted(zip(importances, feature_list), reverse=True)

#create two lists from the previous list of tuples
df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
importance= list(df['importance'])
feature= list(df['feature'])

#see df
print(df)


#Test set rmse and accuracy
from sklearn.metrics import mean_squared_error
final_model = grid_search_forest.best_estimator_
# Predicting test set results
final_pred = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_pred)
final_rmse = np.sqrt(final_mse)
print('The final RMSE on the test set is', round(final_rmse, 2))

#calculate accuracy
errors = abs(final_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))
# Calculate and display accuracy
accuracy = 100 - mape    
#print result
print('The best model achieves on the test set an accuracy of', round(accuracy, 2),'%')


