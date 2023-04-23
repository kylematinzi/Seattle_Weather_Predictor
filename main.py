# categorical data. Trying to predict with the independent variables (temperatures, wind speed, ...) what the
# the dependent variable will be (weather condition).

import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics
from sklearn.linear_model import LogisticRegression

dataFrame = pd.read_csv('seattle-weather.csv')

# this will show 4 graphs; describes the data by showing the distribution of the individual features
dataFrame.hist()
pyplot.show()

# this will show scatter plot distributions of the features;
# run the program in python console mode in order to display in pycharm
# right click, run in console
# scatter_matrix(dataFrame)

# Need to increase the maximum amount of iterations in order to go through the complete dataset
myData_model = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
# dependent variable is "y" weather condition
y = dataFrame.values[:, 5]
# independent variable is "x" weather features
# intentionally leaving out first line which is the date for simplicity
x = dataFrame.values[:, 1:5]

# Training the model
myData_model.fit(x, y)
# User entry data: Inches of precipitation, Max Temperature, Min Temperature, Wind speed Mph

# Test Data cases (4):

# Test #1: 1/8/2012,0,10,2.8,2,sun
# print("Should predict: Sun")
# print(myData_model.predict([[0, 10, 2.8, 2]]))

# Test #2: 1/10/2012,1,6.1,0.6,3.4,rain
# print("Should predict: Rain")
# print(myData_model.predict([[1, 6.1, 0.6, 3.4]]))

# Test #3: 1/19/2012,15.2,-1.1,-2.8,1.6,snow
# print("Should predict: Snow")
# print(myData_model.predict([[15.2, -1.1, -2.8, 1.6]]))

# Test #4: 3/26/2012,0,12.8,6.1,4.3,drizzle
# print("Should predict: Drizzle")
# print(myData_model.predict([[0, 12.8, 6.1, 4.3]]))


# Measure accuracy of model
# y_pred = myData_model.predict(x)
# print("Accuracy: ", metrics.accuracy_score(y, y_pred))

# Shows accuracy of each prediction
# metrics.plot_confusion_matrix(myData_model, x, y)

# Machine learning applied in models
# 3 pictures data show distribution, scatter plot, accuracy of prediction
# Can be used by user, enters weather features

# User interface
precipitationInput = input("Enter inches of precipitation: ")
maximumTempInput = input("Enter maximum temperature: ")
minimumTempInput = input("Enter minimum temperature: ")
windSpeedInput = input("Enter wind speed: ")

precipitation = int(precipitationInput)
maximumTemp = int(maximumTempInput)
minimumTemp = int(minimumTempInput)
windSpeed = int(windSpeedInput)

print(myData_model.predict([[precipitation, maximumTemp, minimumTemp, windSpeed]]))





