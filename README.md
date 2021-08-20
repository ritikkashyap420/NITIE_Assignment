# NITIE_Assignment
Create Fraud-Risk Models to reduce the fraud in e-commerce market.
Dataset used: https://www.kaggle.com/mlg-ulb/creditcardfraud

Here, I used different machine learning and deep learning algorithms to solve the given problem.

1) **EDA + XGboost**: 
 - Here, we first processed the data and removed the columns which were not of much use e.g Time.
 - Now, we can see that the given data set is highly imbalanced, this type of data can cause some issue while creating the model. So, we used SMOTE here(will se in detail in 3rd python file)
 - Used XGB classifier along with the GridSeachCV (which is used for hyperparameter tuning).
 
2) **Light GBM**: 
 - LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
  (Faster training speed and higher efficiency, Lower memory usage, Better accuracy).
 - Normlalised the amount column in the dataset and removed the columns which were not of much use e.g Time.
 - Implemented the LGBM classifier on the dataset without any hyperparameter tuning.
 - Used RandomSearchCV for the hyperparameter tuning and reimplemented the model with best parameters.

I read this paper for reference: http://www.techscience.com/cmc/v61n1/23107/pdf
 
3) **Sampling + Neural Network**
