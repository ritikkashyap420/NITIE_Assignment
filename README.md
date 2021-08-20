# NITIE_Assignment
Create Fraud-Risk Models to reduce the fraud in e-commerce market.
Dataset used: https://www.kaggle.com/mlg-ulb/creditcardfraud

Here, I used different machine learning and deep learning algorithms to solve the given problem.

1) **EDA + XGboost**: 
 - Here, we first processed the data and removed the columns which were not of much use e.g Time.
 - Now, we can see that the given data set is highly imbalanced, this type of data can cause some issue while creating the model. So, we used SMOTE here(will se in detail in 3rd python file)
 - Used XGB classifier along with the GridSeachCV (which is used for hyperparameter tuning).
 
2) **Light GBM**: 
 LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages    (Faster training speed and higher efficiency, Lower memory usage, Better accuracy).
 - Normalised the amount column in the dataset and removed the columns which were not of much use e.g Time.
 - Implemented the LGBM classifier on the dataset without any hyperparameter tuning.
 - Used RandomSearchCV for the hyperparameter tuning and reimplemented the model with best parameters.

 I used this paper on LGBM for fraud detection (for reference): http://www.techscience.com/cmc/v61n1/23107/pdf
 
3) **Sampling + Neural Network**: Since, we faced the issue of Imbalanced dataset now we will try to tackle it with different sampling methods and also see the performance by implementing Random Forest Classifier.
 - Under Sampling: Randomly duplicate examples in the minority class.
 - Over Sampling: Randomly duplicate examples in the majority class.
 - SMOTE: A method that instead of simply duplicating entries creates entries that are interpolations of the minority class, as well as undersamples the majority class.
 - SMOTETomek is a hybrid method which is a mixture of the above two methods, it uses an under-sampling method (Tomek) with an oversampling method (SMOTE). This is present within imblearn. combine module.
- Implemented a simple and weighted neural network with binary CrossEntropy to classify the result.
