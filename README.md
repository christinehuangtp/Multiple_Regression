# Multiple_Regression
In this assignment, you will perform multiple regression analysis on a dataset. Will use a linear model, try to find data where the
dependent variable is linearly dependent on at least two independent variables.

### Student Achievement
#### Kaggle dataset : https://www.kaggle.com/dipam7/student-grade-prediction
This data is about student performance in dairly life and thier grade. There are 395 rows in the data. We are interested in whether the student behaviors have correlation with their grade. In the experiment, we set:

- Dependent (y) = 3nd period grade

- Independent (X) = studytime(X1) ,failures (X2), familyrelationship(X3), absences(X4) ,1st period grade (X5), 2nd period grade(X6)

#### Explore the data
- data.head()
- data.describe()

#### Cleanup and Transform the data
Extract the data with specific columns that we need for analysis

#### Detect the missing data
data_1.isnull().sum()

#### Rename the column names
data_1.columns = ["studytime","failures....]

#### Remove the outliers to make it more tidy
- Q1 = data_1.quantile(0.25)
- Q3 = data_1.quantile(0.75)
- IQR = Q3 - Q1

### Analysis
#### Function: graph the scatter plot
- import matplotlib.pyplot as plt
- import seaborn as sns
Scatterplots of each independent variable vs. the dependent variable (3rd period grade)
                   
#### Split the dataset to train and test dataset
- import numpy as np
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(data_1.drop("3rd period grade", axis = 1), data_1["3rd period grade"], random_state=11)

#### Build the multiple linear regression model with train dataset
- from sklearn.linear_model import LinearRegression
- linear = LinearRegression()
- linear.fit(X=X_train, y=y_train)
- coeff = linear.coef_
- intercept = linear.intercept_

#### Test the model with X_test and print out the accuracy rate of prediction
- import numpy as np
- predicted = np.array(linear.predict(X_test))
- expected  = np.array(y_test)

#### Compute the coefficient of determination and correlation coefficient
- import math
- from sklearn import metrics
- r2 = metrics.r2_score(expected, predicted)
- r  = math.sqrt(r2)

#### Strong linear correlation
- coefficient of determination = 0.86
- correlation coefficient = 0.93

#### Graph the scatter plot and regression line of expected y value and predicted y value
Two common problems that can prevent a machine learning model from making accurate predictions.
- Underfitting : The model is too simple to make predictions, based on its training data
- Overfitting : The model is too complex.
