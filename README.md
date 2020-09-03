Project Scope

   The goal of this project is to predict the chances of certification for an H1-B Visa applicant based on their application data. I determined this issue to be a binary classification problem, where the end result should accurately, within certain limits, predict either a “CERTIFIED” or “DENIED” label based on other relevant information. The logistic regression model attempts to learn the correlationbetween input attributes and a label. Logistic regression is a popular binary classifier. In this case the input attributes with be the applicant’s information and the binary label will be the final case status. The dataset used in this project is the H-1B Visa Petitions 2011-2016 dataset from Kaggle in (.csv) format, with a total of (3,002,458) applications.
 
 
Experimental Results and Analysis 

   I used the ChiSqSelector from the pyspark.ml.feature class to implement chi-squared test and select the top features related to the OUTCOME label. The top 4 categorical features according the the chi squared test were CITY, PREVAILING_WAGE, FULL_TIME_POSITION and SOC_NAME. The other features were discarded and the resulting dataset was passed to a VectorAssembler. The data was randomly split into testing and training data with the dimensions of 0.3 and 0.7. I adjusted the logistic regression parameters elasticNetParam and regParam while keeping the maxIterations at 15 constant. The results of those models are listed below. After phases of retraining, the best-performing model on the test dataset was the logistic regression model with regParam 0.01, elasticNetParam 0.15, with an intercept of 4.415. The evaluation of the model was done using the ROC and PR metrics, with a best score of 0.7012 and 0.9899 respectively.. As predicted, the PR metric scored the models more favorably than ROC since it handles unbalanced data. The results of those metrics are also listed below.

   The results showed that the most predictive feature is the city where the job will be located. The other top three features were the salary, if the job was a full time position, and the job type. The top 4 features from the chiSqSelector were used in the model, which predicted the case status with an average accuracy of [insert] by the PR AUC. Further analysis of the features, with time permitting, could produce a better model, but overall, I am satisfied with the model’s performance based on the PR AUC metric.


How To Run

1. Download the H1B Visa Applications dataset from Kaggle. Save dataset as h1b_kaggle.csv.
    https://www.kaggle.com/nsharan/h-1b-visa
2. Run preprocess.py, model.py, and demo.py in order.

Dependencies

Python 2.7,
Pyspark 2.0 or greater
