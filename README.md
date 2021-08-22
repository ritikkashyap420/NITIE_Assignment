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
 
4) **ANN +  Weighted Neural Network**
- Implemented a simple neural network with binary CrossEntropy to classify the result.
- Since, the data is highly imbalanced so we have to intialise weight accordingly
- Then Implemented a weighted neural network and compare the performance of Simple ANN and weightd ANN.

5) **Anomaly Detection using LOF and isolation forest.**
 - Implemented Isolation forest, LOF and SVC and comapred the model performance.
 - Isolation Forest has a 99.74% more accurate than LOF of 99.65% and SVM of 70.09
 - When comparing error precision & recall for 3 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is    around 27 % versus LOF detection rate of just 2 % and SVM of 0%.
 - Used a Turkish Journal as Reference for the task. Link: https://turcomat.org/index.php/turkbilmat/article/view/7473/5991
