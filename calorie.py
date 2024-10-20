


import pandas as pd
calories = pd.read_csv('calories.csv')
exercise= pd.read_csv('exercise.csv')
import pickle
calories.head()
exercise.head()
print(calories.shape)
print(exercise.shape)
print(calories.columns)
print(exercise.columns)
print("\nCalories Data Info:")
print(calories.info())
print("\nExercise Data Info:")
print(exercise.info())
print("\nMissing Values in Calories Data:")
print(calories.isnull().sum())
print("\nMissing Values in Exercise Data:")
print(exercise.isnull().sum())
merged_data = exercise.merge(calories, on='User_ID')
summary_stats = merged_data.describe()
print(summary_stats)
data_info = merged_data.info()

missing_values = merged_data.isnull().sum()
print(missing_values)
gender_counts = merged_data['Gender'].value_counts()
print(gender_counts)
from scipy.stats import zscore
merged_data['Calories_zscore'] = zscore(merged_data['Calories'])
outliers_zscore = merged_data[merged_data['Calories_zscore'].abs() > 3]
print(outliers_zscore)
Q1 = merged_data['Calories'].quantile(0.25)
Q3 = merged_data['Calories'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = merged_data[(merged_data['Calories'] < (Q1 - 1.5 * IQR)) | (merged_data['Calories'] > (Q3 + 1.5 * IQR))]
print(outliers_iqr)
merged_data['Calories_zscore'] = zscore(merged_data['Calories'])
threshold = 3
outliers_zscore = merged_data[merged_data['Calories_zscore'].abs() > threshold]
cleaned_data = merged_data[merged_data['Calories_zscore'].abs() <= threshold]
cleaned_data = cleaned_data.drop(columns=['Calories_zscore'])
print(cleaned_data)df=exercise.merge(calories,on='User_ID')
df.head()
df['Gender']=df['Gender'].map({'male':1,'female':0})
df.head()
import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
lr_model = LinearRegression()
dtr_model = DecisionTreeRegressor()
rfr_model = RandomForestRegressor()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
print("XGBoost Regressor:")
print(f"Mean Squared Error (MSE): {xgb_mse}")
print(f"R² Score: {xgb_r2}\n")xgb_model = xgb.XGBRegressor()
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
import pickle
pickle.dump(xgb_model ,open('xgb_model .pkl','wb'))
X_train.to_csv('X_train.csv')


