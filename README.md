# Auto-Insurance-Claim-Predictor
Python ML project using scikitlearn,pandas,numpy,matplotlib

## Problem Statement:

Build a Machine learning model to predict the probability that owner will initiate an auto insurance claim in the next year.

Most companies charge a flat premium to the customers irrespective of their risk for filing an insurance claim. Inaccuracies in the insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones. Our project will help the insurance company in following ways:

Affluent customer: Company can attract the good drivers if it is doing the fair pricing. Loss ratio: Company can avoid specific customer/policies if they are at high risk of filing claim which in turn decrease loss ratio. Fair pricing: Company can charge the premium to the customers by their risk, and accurate prediction will allow them to tailor their prices further. Claim forecast: Claim is proportional to the number of risky customers, so company forecast the number of claims it could get next year which will help them to manage their fund better

As per industry estimate 1% reduction in the claim can boost profit by 10%. So, through the ML model, we can identify and deny the insurance to the driver who will make a claim.  Thus, ensuring reduced claim outgo and increased profit.

# Analysis Approach

## Exploratory Data Analysis
Contains the initial exploration of data like 
*   Load the Data into DataFrame
*   Inference about the data
*   Finding the distribution of target variables
*   Split features into Categorical,Binary,Ordinal,Interval
*   Handled Missing Values
*   Found Outliers in continuous variables
*   Dropped Id column
*   Inferences about continuous,ordinal,binary variables through Visualization
*   Level of Correlation for interval,ordinal 
*   Keep only 1 column for columns having high correlation
*   Balanced the data by OverSampling(Smote algorithm)
*   One-hot encoding/dummification of the categorical variables
*   Saved CSV Files for data with and without Encoding

## Modeling
* Classification algorithms Like 
  * Logistic Regression
  * Support Vector Machines(Linear,Poly)
  * KNN
  * Decision Trees
  * Random Forest,
  * AdaBoost,XGBoost,GradientBoost are used to build the model
* There is same accuracy for tree based models with and without OneHotEncoding categorical features while accuracy improved for rest
* Ensembling models like Boosting and RandomForest gave the best results among all the algorithms used
