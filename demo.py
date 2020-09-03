# PHASE 3 DEMO THE MODEL TO MAKE PREDICTION


from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext 
from pyspark.sql import SparkSession 
from pyspark.ml.util import MLWriter, MLReader
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Demo Model").getOrCreate()

sqlContext = SQLContext(spark)

#Load file
#Should be some file on your system
print("Creating dataframe...")
sample = sqlContext.read.load('sampledata.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')
						  
						  				  

#Vector assembler to format date for ML algorithms
from pyspark.ml.feature import VectorAssembler
print("Creating feature vector for logistic regression")
newVecAssembler = VectorAssembler(inputCols=['CITY_index','FULL_TIME_POSITION_index', 'EMPLOYER_NAME_index','PREVAILING_WAGE', 'SOC_NAME_index'], outputCol="features")
sample = newVecAssembler.transform(sample)
sample = sample.withColumnRenamed("OUTCOME", "label")

print("Loading logistic regression model...")						  

lr = LogisticRegression(maxIter=15, regParam=0.01, elasticNetParam = 0.15)
[sample1, test_data] = sample.randomSplit([0.9, 0.1])
# Fit the model
lrModel = lr.fit(sample1)

print("Schema of dataframe: ")
sample.printSchema()


predictions = lrModel.transform(test_data)


print("Displaying predictions...")
predictions.show(100)


evaluator = BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
accuracy = evaluator.evaluate(predictions)
print("\n\n\nAccuracy of model: ")
print(accuracy)
