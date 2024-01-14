


#Task: Predict car price using linear regression
#You are provided with a dataset about cars in the form of a csv file. The data contains different details like name, year and other
#You are provided with the code to download the csv file that contains the dataset.
#Your task is to train a linear regression model and predicts the car price.
#Divide the data into a train (80%) and a validation data set (20%).

#Print train and validation losses.
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

!gdown 1dail55JlMcsOlZKiSQeZSbTNHYiBH3U9

"""**Preprocess and clean the data**"""

data = pd.read_csv('car_dataset.csv')
data

"""**Convert strings to numbers**"""

column_to_exclude = "name"
for column in data.columns:
    if column != column_to_exclude:  # Exclude the specified column
        if data[column].dtype == 'object':  # Check if the column contains string/object data
            label_encoder = LabelEncoder()
            data[column] = label_encoder.fit_transform(data[column])

data

"""**Split the data**"""

x_values = data.drop("selling_price", axis=1)
x_values = data.drop("name", axis=1)

y_values = data["selling_price"]
price_column = "selling_price"

x_values

y_values

"""**Split the data into training and testing**"""

X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

"""**Use the linear regression closed-form solution and calculate the Theta**"""

def calculate_theta(X, y):
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta


theta = calculate_theta(X_train, y_train)

theta = theta.reshape(-1, 1)

theta

theta.shape

predections = X_test @ theta

"""**Define the mean-squared function**"""

def mean_squared_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)

predections.shape

result = mean_squared_error(y_test, predections.values.flatten())

print(result)

predections = X_train @ theta
result_with_X_train = result = mean_squared_error(y_train, predections.values.flatten())
print(result_with_X_train)

