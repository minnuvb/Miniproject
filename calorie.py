import pandas as pd
calories = pd.read_csv('calories.csv')
exercise= pd.read_csv('exercise.csv')
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
import matplotlib.pyplot as plt
merged_data.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
plt.show()
plt.figure(figsize=(10, 6))
merged_data.boxplot(column=['Calories'])
plt.title('Box Plot of Calories Burned')
plt.show()





     


     










     

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
