# PHASE 1: INITIAL PRE-PROCESSING OF THE DATA


from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext 
from pyspark.sql import SparkSession 


spark = SparkSession.builder.appName("Python Spark Binary Classifier").getOrCreate()

sqlContext = SQLContext(spark)

#Load file
#Should be some file on your system
data_path = "h1b_kaggle.csv"
data = sqlContext.read.load(dath_path, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')


#Drop the columns we dont need (i.e geographical location data that is not discrete)
data = data.drop("_c0", "YEAR", "lon", "lat")


#Remove the case statuses that are not explicitly certified or denied
data = data[data['CASE_STATUS'] != 'REJECTED']  
data = data[data['CASE_STATUS'] != 'WITHDRAWN']  
data = data[data['CASE_STATUS'] != 'CERTIFIED-WITHDRAWN']
data = data[data['CASE_STATUS'] != 'INVALIDATED']  
data = data[data['CASE_STATUS'] != 'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']  
data = data[data['CASE_STATUS'] != 'NA']  

data.groupBy('CASE_STATUS').count().show()

#Split the location column into two parts, city and state
split_col = split(data['WORKSITE'], ', ')
data = data.withColumn('CITY', split_col.getItem(0))
data = data.withColumn('STATE', split_col.getItem(1))
data = data.drop(data.WORKSITE)



#Add an outcome column to use for the model
data = data.withColumn("OUTCOME",
  when(col("CASE_STATUS") == "CERTIFIED", 1.0).otherwise(0.0))

#Cast the wage as an integer
data2 = data.select('EMPLOYER_NAME', 'SOC_NAME', 'FULL_TIME_POSITION', data.PREVAILING_WAGE.cast("integer").alias('PREVAILING_WAGE'), 'CITY', 'STATE', 'OUTCOME')


# Get the average wage for each profession and join it to the original table doing a inner join
data3 = data2.groupby('SOC_NAME').mean('PREVAILING_WAGE')
data3 = data2.join(data3, ['SOC_NAME'], 'inner')

#Replace the previous wage column with the average wage column
data3 = data3.drop("PREVAILING_WAGE")
data2 = data3.withColumnRenamed('avg(PREVAILING_WAGE)', 'PREVAILING_WAGE')


#Add a count of how many applications each employer has
data2 = data2.join(data2.groupBy('EMPLOYER_NAME').count(),on='EMPLOYER_NAME')
data2 = data2.withColumnRenamed('count', 'EMPLOYER_APP_COUNT')

# Filter rows with applications less than 4 to clean data
data2 = data2.filter(data2.EMPLOYER_APP_COUNT > 5)


#Add a count of how many applications each SOC_NAME has
data2 = data2.join(data2.groupBy('SOC_NAME').count(),on='SOC_NAME')
data2 = data2.withColumnRenamed('count', 'SOC_NAME_COUNT')

# Filter rows with applications less than 4 to clean data
data2 = data2.filter(data2.SOC_NAME_COUNT > 5)


data2.printSchema()

data2.coalesce(1).write.save("h1bclean", format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')

spark.stop()


