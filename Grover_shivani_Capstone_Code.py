# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:58:21 2019

@author: Shivani Grover
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
pd.options.display.max_columns = None

#The link from where the dataset has been extracted 
#https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

#### LOADING THE DATASET
wine = pd.read_csv("C:/Users/17815/Documents/AST/capstone/winequality-red.csv")
print(wine.head(5))

########### Describing the dataset
print(wine.describe())
print(wine.info())
######  Getting all the independent variables
wine_iv = wine.iloc[ : , 0:11]

##########  Naming the columns to avoid whitespace
wine.columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol","quality"]
wine_iv.columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]


###############################   EDA   ########################################

#Finding the null values

null_counts = wine.isnull().sum()
print(null_counts[null_counts > 0].sort_values(ascending=False))  # 7% null
null_rows = wine.isnull().any(axis=1)
wine_null = wine[null_rows]  # 5% of the total 1599 rows are null
len(wine_null)
wine_not_null = wine.dropna()  #### dropping all the missing records
len(wine_not_null)

##########     Outlier Analysis    ############
q1 = wine_iv.quantile(0.25, axis = 0)
q3 = wine_iv.quantile(0.75, axis = 0)
IQR = q3 - q1
limit_max = q3 + 1.5*IQR
limit_min = q1 - 1.5*IQR 
outliers = (wine_iv < limit_min) | (wine_iv > limit_max)
wine_out = wine[~outliers.any(axis=1)]  ### Outliers removed
print(len(wine_out))  ########## Number of rows after removing our=tliers

####   CITRIC ACID (OUTLIER ANALYSIS) ####
####  working with null values
ca = wine["citric_acid"]
ca_not_null = ca.dropna()   #### data series with no missing value
ca_filled = ca.fillna(0)    #### replacing the null values with 0
a = sns.distplot(ca_not_null, color = "red")
a.set_title("Histogram aftering removing missing values")
a.axes.xaxis.label.set_text("Citric acid")
a.axes.yaxis.label.set_text("Counts")

b = sns.distplot(ca_filled, color = "green")
b.set_title("Histogram after replacing the missing values")
b.axes.xaxis.label.set_text("Citric acid")
b.axes.yaxis.label.set_text("Counts")

######  removing the outliers and replacing the rest missing values
ca_oa_not_null = wine_out["citric_acid"].dropna()   ####  data series  with no missing value
ca_oa_filled = wine_out["citric_acid"].fillna(0)    #### replacing the null values with 0
c = sns.distplot(ca_oa_not_null, color = "black")
c.set_title("Histogram without outliers and missing values")
c.axes.xaxis.label.set_text("Citric acid")
c.axes.yaxis.label.set_text("Counts")

d = sns.distplot(ca_oa_filled, color = "blue")
d.set_title("Histogram without outliers and filled missing values")
d.axes.xaxis.label.set_text("Citric acid")
d.axes.yaxis.label.set_text("Counts")

#### Comparing distribution before and after removing outliers
a = sns.distplot(ca_not_null, color = "red")
a.set_title("Histogram aftering removing missing values")
a.axes.xaxis.label.set_text("Citric acid")
a.axes.yaxis.label.set_text("Counts")
c = sns.distplot(ca_oa_not_null, color = "black")


#####  FIXED ACIDITY (OUTLIER ANALYSIS) #####
fa = wine["fixed_acidity"]
fa_not_null = fa.dropna()
fa_filled =  fa.fillna(fa.mean())
e = sns.distplot(fa_not_null, color = "red")
e.set_title("Histogram without missing values")
e.axes.xaxis.label.set_text("Fixed acidity")
e.axes.yaxis.label.set_text("Counts")

f = sns.distplot(fa_filled, color = "green")
f.set_title("Histogram after replacing the missing values")
f.axes.xaxis.label.set_text("Fixed acidity")
f.axes.yaxis.label.set_text("Counts")

######  removing the outliers and replacing the rest missing values
fa_oa_not_null = wine_out["fixed_acidity"].dropna()
fa_oa_filled = wine_out["fixed_acidity"].fillna(wine_out["fixed_acidity"].mean())

g = sns.distplot(fa_oa_not_null, color = "black")
g.set_title("Histogram without outliers and missing values")
g.axes.xaxis.label.set_text("Fixed acidity")
g.axes.yaxis.label.set_text("Counts")

h =sns.distplot(fa_oa_filled, color = "blue")
h.set_title("Histogram without outliers and filled missing values")
h.axes.xaxis.label.set_text("Fixed acidity")
h.axes.yaxis.label.set_text("Counts")

#### Comparing distribution before and after removing outliers
e = sns.distplot(fa_not_null, color = "red")
e.set_title("Histogram without missing values")
e.axes.xaxis.label.set_text("Fixed acidity")
e.axes.yaxis.label.set_text("Counts")
g = sns.distplot(fa_oa_not_null, color = "black")

######## DATA IMPUTATION  ##########
#### creating a new dataframe to work on 
wine_new = wine

## grouping the data set by quality rating
groups = wine_new.groupby("quality").aggregate([min, max, np.mean])
print(groups)

## filling the missing records with mean or 0
acid = wine_new["citric_acid"].fillna(0)
wine_imputed = wine_new.groupby("quality").transform(lambda x: x.fillna(x.mean()))
wine_imputed["citric_acid"] = acid    
wine_full = wine_imputed
wine_full["quality"] = wine_new["quality"]
print(wine_imputed.isnull().sum())


####creating a count plot
fig = plt.figure(figsize = (5,5))
b = sns.countplot(wine_full["quality"])
b.set_title("Number of Records corresponding Quality")
#####getting a correlation coefficient
wine_full.corr()

###### finding distributions for each varable
fig = plt.figure(figsize = (15,25))
j = 0
for i in wine_full.columns:
    fig.add_subplot(11, 1, j+1)
    j += 1
    k = sns.distplot(wine_full[i])
    k.set_title(i)
    
###########   MODELLING   ################

#  Creating train and test dataframes
X_train, X_test, y_train, y_test = train_test_split(wine_imputed, wine_new["quality"], test_size=0.20)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

### Normalization
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit(X_train)  # .fit() To calculate the mean and std values to z-normalize the dataset
X_train = scaler.transform(X_train)   # transform() is used to returna  dataframe with the z-normalized values
X_test = scaler.transform(X_test)   # .fit_transform() is a one step function to fit and transform the data frame values

######### KNN  ###########
KNN = knn(n_neighbors = 20)
KNN.fit(X_train, y_train)
Prediction_knn = KNN.predict(X_test) 
print(confusion_matrix(y_test, Prediction_knn))
print(classification_report(y_test, Prediction_knn))


##########  GAUSSIAN NAIVE BAYES  #########
clf = GaussianNB()
nb = clf.fit(X_train, y_train)
Prediction_nb = nb.predict(X_test)
print(confusion_matrix(y_test, Prediction_nb))
print(classification_report(y_test, Prediction_nb))

##########  Random forest Classifier  #########
rf = RandomForestClassifier() 
rf = rf.fit(X_train,y_train) 
Prediction_rf = rf.predict(X_test) 
print(confusion_matrix(y_test, Prediction_rf))
print(classification_report(y_test, Prediction_rf))


