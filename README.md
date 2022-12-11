# Auto-Insurance-Claim-Predictor
Python ML project using scikit learn,pandas,numpy,matplotlib

## Problem Statement:

To Predicts the probability that a driver will initiate an auto insurance claim in the next year.

Most companies charge a flat premium to the customers irrespective of their risk for filing an insurance claim. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones. Our project will help the insurance company in following ways:

Affluent customer: Company can attract the good drivers if it is doing the fair pricing. Loss ratio: Company can avoid specific customer/policies if they are at high risk of filing claim which in turn decrease loss ratio. Fair pricing: Company can charge the premium to the customers by their risk, and accurate prediction will allow them to tailor their prices further. Claim forecast: Claim is proportional to the number of risky customers, so company forecast the number of claims it could get next year which will help them to manage their fund better

# Analysis Approach

## Exploratory Data Analysis
Contains the initial exploration of data like 
* Finding the distribution of target variables
* Handling missing values
* Inferences about continuous,ordinal,binary variables through Visualization
* Level of correlated variables
* Balancing the data using SMOTE 
* One-hot encoding/dummification of the categorical variables
* Feature scaling-Standardization for all ordinal and interval columns 

## Modeling
All Classification algorithms Like Logistic Regression,Support Vector Machines(Linear,Poly),KNN,Decision Trees,Random Forest,AdaBoost,XGBoost,GradientBoost
are used
Ensembling model like RandomForest and Bossting gave best results among all the algorithms
