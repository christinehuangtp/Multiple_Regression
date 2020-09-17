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
import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(df):
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    for col in df.drop("3rd period grade", axis = 1).columns:
        plt.figure(figsize=(16, 9))
        sns.scatterplot(data=df, x=col, y='3rd period grade', 
                     hue='3rd period grade', 
                     palette='cool', legend=False)
                    
#### Split the dataset to train and test dataset

#### Build the multiple linear regression model with train dataset

#### Test the model with X_test and print out the accuracy rate of prediction

#### Compute the coefficient of determination and correlation coefficient

#### Strong linear correlation
