# PHASE 2: CREATE THE MODEL

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext 
from pyspark.sql import SparkSession 
from pyspark.ml.util import MLWriter, MLReader
from pyspark.ml.classification import LogisticRegressionModel

spark = SparkSession.builder.appName("Build Model").getOrCreate()

sqlContext = SQLContext(spark)

#Load file
#Should be some file on your system
data_path = "h1b_kaggle.csv"
data2 = sqlContext.read.load(data_path, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

#Pipeline to transform strings into numerical categories
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer


indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").setHandleInvalid("skip").fit(data2) for column in list(set(data2.columns)-set(['OUTCOME'])) ]


pipeline = Pipeline(stages=indexers)
data2 = pipeline.fit(data2).transform(data2)



data3 = data2.select('OUTCOME', 'CITY_index','FULL_TIME_POSITION_index', 'PREVAILING_WAGE_index','EMPLOYER_NAME_index',	'SOC_NAME_index',	'STATE_index','SOC_NAME_COUNT_index','EMPLOYER_APP_COUNT_index')
data3 = data3.na.drop()
data3 = data3.drop("EMPLOYER_NAME_index")


#Vector assembler to format date for ML algorithms
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=['CITY_index','FULL_TIME_POSITION_index', 'PREVAILING_WAGE_index','SOC_NAME_index','STATE_index','SOC_NAME_COUNT_index','EMPLOYER_APP_COUNT_index'], outputCol="features")
data4 = vecAssembler.transform(data3)


# Chi squared selector to select the features that the label depends on

from pyspark.ml.feature import ChiSqSelector
selector = ChiSqSelector(numTopFeatures=4, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="OUTCOME")
result = selector.fit(data4).transform(data4)
print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()


#Use top features to make new vector for linear regression
newVecAssembler = VectorAssembler(inputCols=['CITY_index','FULL_TIME_POSITION_index', 'EMPLOYER_NAME_index','PREVAILING_WAGE', 'SOC_NAME_index'], outputCol="features")
newdata = data2.select('OUTCOME', 'CITY_index','FULL_TIME_POSITION_index', 'EMPLOYER_NAME_index','PREVAILING_WAGE','SOC_NAME_index')
newdata1 = newVecAssembler.transform(newdata)


#Logistic Regression
#Split the new data in test and train
newdata1 = newdata1.withColumnRenamed("OUTCOME", "label")
[test_data, train_data] = newdata1.randomSplit([0.3, 0.7])


'''
data4 = data4.withColumnRenamed("OUTCOME", "label")
data4 = data4.na.drop()
data5 = data4.select("label", "SOC_NAME_index", "FULL_TIME_POSITION_index", "PREVAILING_WAGE_index", "CITY_index","features")
[test_data, train_data] = data5.randomSplit([0.3, 0.7])
'''

# Build Model

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=15, regParam=0.0001, elasticNetParam = 0.15)

# Fit the model
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#Evaluate accuracy of the model

from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions = lrModel.transform(test_data)
evaluator = BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
accuracy = evaluator.evaluate(predictions)
print("The Model accuracy is: ")
print(accuracy)
  
#Save Model and data for demo
sample = newdata.sample(False, 0.01, 42).limit(1000000)


sample.coalesce(1).write.save("demo_data", format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')


