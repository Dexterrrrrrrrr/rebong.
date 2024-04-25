# Activities and Project Compilation

# Introduction to Compilers: A Key Element in Computing and Beyond
Compilers are essential tools in computer science, data science, machine learning, and qualitative methods. They translate high-level programming languages into machine code, enabling efficient execution of algorithms and programs. In computer courses, compilers teach language design and optimization. In data science, they optimize data processing tasks for faster insights extraction. In machine learning, they accelerate model deployment across diverse architectures. In qualitative methods, compilers automate analysis tasks, enhancing research efficiency. Overall, compilers empower innovation and efficiency across disciplines, bridging the gap between human-readable code and machine-executable instructions.

# What is Importing Libraries
Importing libraries in Google Colab works similarly to how it works in any Python environment. You can import libraries using the import statement, and Colab provides a convenient environment where you can install and import libraries on-the-fly.

# Examples of Importing Libraries
import pandas as pd #Data Manipulation
import numpy as np # Numerical operation
import matplotlib.pyplot as plt #Data Visualization
from sklearn.linear_model import LinearRegression #Data Modeling

# What is Loading dataset
Loading a dataset refers to the process of reading data from a file or an external source into your program or environment for analysis, manipulation, or further processing. It's a crucial step in data analysis, machine learning, and many other data-related tasks
# Here's the example of Loading dataset
data = pd.read_csv('sales_data_2.csv', encoding = "latin-1")
data.head()

# Data Preprocessing
Load the Dataset: Reads the dataset from a CSV file called "Netflix_Userbase.csv" into a Pandas DataFrame named data.
Handling missing values: Prints the sum of missing values for each column in the dataset.
Encode Categorical Variables: Creates dummy variables for categorical features using one-hot encoding and converts the 'Monthly Revenue' column into a binary variable indicating whether the revenue is greater than 10.
Feature Selection: Prepares the feature matrix X and the target variable y by dropping irrelevant features ('Monthly Revenue' and 'Age') from the dataset.

# Exploratory Data Analysis (EDA)
Descriptive Statistics: Computes and prints descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max) for the feature matrix X.
Visualizations: Plots histograms to visualize the distributions of 'Monthly Revenue' and 'Age' in the dataset.

# Linear Regression Model (Predicting Monthly Revenue)
Build the Model: Splits the data into training and testing sets, fits a linear regression model to the training data, and predicts the target variable ('Monthly Revenue') on the test set.
Model Evaluation: Calculates evaluation metrics (RMSE and R-squared) to assess the performance of the linear regression model.

# Logistic Regression Model (Predicting Customer Feedback)
Model Building: Creates a binary target variable 'Feedback' based on whether 'Monthly Revenue' is above the mean revenue. Splits the data into training and testing sets and fits a logistic regression model to predict 'Feedback'.
Model Evaluation: Computes evaluation metrics (accuracy, precision, recall, F1 score, and confusion matrix) to evaluate the performance of the logistic regression model.

# Comparative Analysis and Visualization
Linear Regression Feature Importance: Fits a linear regression model to the entire dataset to determine the coefficients of each feature and visualizes the feature importance.
Logistic Regression Feature Importance: Uses a pipeline with preprocessing steps (scaling numerical features and one-hot encoding categorical features) to fit a logistic regression model and visualize the feature importance.

# Generating Random Data:
Generates 50 random data points for the independent variable X ranging from 0 to 100.
Computes the dependent variable Y as a linear function of X plus some normally distributed noise.
Creates a DataFrame data to store the generated data.
Saves the DataFrame as a CSV file named 'data.csv'.

# Visualizing the Data:
Plots a scatter plot of the generated data points with X on the x-axis and Y on the y-axis.

# Preparing Data for Modeling:
Separates the independent variable X and dependent variable Y from the DataFrame.

# Fitting the linear regression model:
Initializes a linear regression model.
Fits the model to the data.

# Getting model coefficients:
Retrieves the slope (coefficient) and intercept of the fitted regression line.

# Visualizing the linear regression line:
*Plots the original data points.
Plots the regression line based on the fitted model.*

# Making Prediction:
*Prompts the user to input a value for X.
Uses the trained model to predict the corresponding Y value for the input X.
Prints the input X value and the predicted Y value.*
