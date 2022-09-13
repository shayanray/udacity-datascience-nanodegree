#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project Workspace
# This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.
# 
# You can follow the steps below to guide your data analysis and model building portion of this project.

# In[1]:


# import libraries
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, avg
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.window import Window
from pyspark.sql import Row

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, PCA, RegexTokenizer, VectorAssembler, Normalizer, StandardScaler


import pandas as pd
import numpy as np
import re, time
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# set display options
pd.set_option('display.max_columns', None)


# In[3]:


# create a Spark session
spark = SparkSession.builder.appName("Udacity-Sparkify-Capstone").getOrCreate()


# In[4]:


spark.sparkContext.getConf().getAll()


# # Load and Clean Dataset
# In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 

# In[5]:


dataset_df = spark.read.json("./mini_sparkify_event_data.json")


# In[6]:


dataset_df.show(5, truncate = False)


# In[7]:


dataset_df.describe()


# In[8]:


dataset_df.printSchema()


# Possible candidate columns for model training
# 
# - userId
# - gender
# - itemInSession
# - length
# - level
# - location
# - page
# - registration
# - artist
# - song
# - userAgent
# 
# 
# Check for isnull for each of these columns to discard empty rows or impute depending on findings
# 
# The following at a first glance do not seem useful to predict churn.
# - auth (being logged in or not)
# - firstname (user's first name)
# - lastname (user's last name)
# 

# In[9]:


dataset_df.filter(  isnull(dataset_df["userId"]) | 
                    isnull(dataset_df["gender"]) | 
                    isnull(dataset_df["itemInSession"]) | 
                    isnull(dataset_df["length"]) | 
                    isnull(dataset_df["level"]) | 
                    isnull(dataset_df["location"]) | 
                    isnull(dataset_df["page"]) | 
                    isnull(dataset_df["registration"]) |
                    isnull(dataset_df["artist"]) | 
                    isnull(dataset_df["song"]) | 
                    isnull(dataset_df["userAgent"]) 
                 ).count()


# In[10]:


dataset_df.take(2)


# In[11]:


dataset_df.count()


# In[12]:


# get basic statistics for artist
dataset_df.describe("artist").show()


# In[13]:


dataset_df.groupby("page").count().show(truncate = False)


# In[14]:


# dataset_df.groupby(["userId"]).sum().show(100, truncate=False)


# # Clean the data

# In[15]:


dataset_df.filter((dataset_df["userId"] == None) | (dataset_df["userId"] == "")).count()


# In[16]:


dataset_df.filter(dataset_df.gender.isNotNull()).count()


# In[17]:


cleaned_df = dataset_df.filter( dataset_df.userId.isNotNull() & 
                                dataset_df.gender.isNotNull() &
                                dataset_df.itemInSession.isNotNull() &
                                dataset_df.level.isNotNull() &
                                dataset_df.location.isNotNull() &
                                dataset_df.page.isNotNull() &
                                dataset_df.registration.isNotNull() &
                                dataset_df.userAgent.isNotNull() 
                )
cleaned_df.count()


# In[18]:


interesting_churn_find_df = dataset_df.filter( dataset_df.userId.isNotNull() & 
                                dataset_df.gender.isNotNull() &
                                dataset_df.itemInSession.isNotNull() &
                                dataset_df.level.isNotNull() &
                                dataset_df.location.isNotNull() &
                                dataset_df.page.isNotNull() &
                                dataset_df.registration.isNotNull() &
                                dataset_df.userAgent.isNotNull() &
                                dataset_df.length.isNotNull() & 
                                dataset_df.artist.isNotNull() &
                                dataset_df.song.isNotNull() 
                )
interesting_churn_find_df.count()


# In[19]:


cleaned_df.filter((dataset_df["userId"] == None) | (dataset_df["userId"] == "")).count()


# In[20]:


cleaned_df.groupby("userId").count().show(truncate = False)


# # Exploratory Data Analysis
# When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.
# 
# ### Define Churn
# 
# Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.
# 
# ### Explore Data
# Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

# In[21]:


create_churn_label = udf(lambda page: 1.0 if page == "Cancellation Confirmation" else 0.0, FloatType())


# In[22]:


cleaned_df = cleaned_df.withColumn("churn", create_churn_label(cleaned_df["page"]))


# In[23]:


interesting_churn_find_df = interesting_churn_find_df.withColumn("churn", create_churn_label(interesting_churn_find_df["page"]))


# In[24]:


dataset_df = dataset_df.withColumn("churn", create_churn_label(interesting_churn_find_df["page"]))


# In[25]:


cleaned_df.head(2)


# In[26]:


churned_count_df = cleaned_df.select(["userId", "churn"]).dropDuplicates().groupby("churn").count()
churned_count_df.show(truncate = False)


# In[27]:


interesting_churn_find_df.select(["userId", "churn"]).dropDuplicates().groupby("churn").count().show(truncate = False)


# In[28]:


# looks like the churn record has length, artist, song, churn as null
cleaned_df.select(["userId", "length", "artist", "song", "churn"]).filter(cleaned_df.churn == 1).show(truncate = False)


# In[29]:


cleaned_df.filter(cleaned_df.churn == 1).show(2, truncate = False)


# ## Churn Distribution Plot

# In[30]:



plt.figure(figsize = [6,4])
churned_count_df.toPandas().plot(kind="bar", x="churn", y="count", title="Churned Users count")
plt.xlabel("churn value")
plt.ylabel("count")
plt.grid(True)
plt.legend(["churned", "not churned"], loc ="upper left")
plt.show()


# This is a clear case of imbalanced dataset and appropriate ML model selection and metrics are required to deal with this situation. TODO add more details

# ## Gender impact on Churning

# In[31]:


num_gender_df = cleaned_df.select([ "gender","userId", "churn"]).dropDuplicates().groupby(["gender","churn"]).count()
num_gender_df.show()
num_gender_df.toPandas().plot(kind="bar", x="gender", y="count")
plt.xlabel("gender")
plt.ylabel("churn-count")
plt.grid(True)
plt.legend(["churned", "not-churned"], loc ="upper left")
plt.show()


# #### Observation from Gender vs Churn
# 
# The graph shows males(61.5%) tend to churn more than females(38.5%).

# ### Level Impact vs Churn 

# In[32]:


num_level_df = cleaned_df.select([ "level","userId", "churn"]).dropDuplicates().groupby(["level","churn"]).count()
num_level_df.show()
num_level_df.toPandas().plot(kind="bar", x="level", y="count")
plt.xlabel("level")
plt.ylabel("churn-count")
plt.grid(True)
plt.legend(["paid", "free"], loc ="upper left")
#plt.show()


# #### Observation from Level/Tier vs Churn
# 
# Graph shows paid tier users tend to churn more(59.6%) than free tier users(40.4%)

# ### Browser/UserAgent Impact on Churn 

# In[33]:


num_useragent_df = cleaned_df.filter(cleaned_df["churn"] == 1).select([ "userAgent","userId", "churn"]).dropDuplicates().groupby(["userAgent","churn"]).count()
num_useragent_df.show(truncate=False)


# In[34]:


num_useragent_df.toPandas().plot(kind="bar", x="userAgent", y="count", figsize = [18,16], legend=True)
plt.legend(["churned", "not-churned"], loc ="upper left")


# #### Observations from useragent
# 
# There is no overwhelming or significant correlation of user-agent to subscription cancellation. 
# (Mac and Windows Safari browser each have 5 cancellations in the highest values)
# Discarding this feature for model prediction.

# In[ ]:





# # Feature Engineering
# Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
# - Write a script to extract the necessary features from the smaller subset of data
# - Ensure that your script is scalable, using the best practices discussed in Lesson 3
# - Try your script on the full data set, debugging your script if necessary
# 
# If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

# #### Useful features for model building
# 
# ##### Categorical:
# 
# - gender
# - level
# 
# 
# ###### Numerical:
# 
# - number of unique songs played per userId
# - number of total songs played per userId
# - number of unique artists per userId
# - number of Ads action per userId
# - number of thumb down action per userId
# - number of thumbs up action per userId
# - number of friends added per userId
# - number of days after initial registration per userId

# #### create feature : gender

# In[35]:


set_gender_value = udf(lambda x: 1.0 if x == 'M' else 0.0, FloatType())


# In[36]:


ftr_gender  = cleaned_df.withColumn("gender_value", set_gender_value("gender")).select(["userId","gender_value"]).dropDuplicates(["userId","gender_value"])
ftr_gender.show(truncate=False)


# #### create feature : last known level (latest_level)  - free or paid

# In[37]:


set_level_value = udf(lambda x: 1.0 if x == 'paid' else 0.0, FloatType())


# In[38]:


ftr_level = cleaned_df.orderBy('ts', ascending=False).groupBy('userId').agg(first('level').alias('latest_level')).drop('level')
ftr_level = ftr_level.withColumn("level_value", set_level_value("latest_level")).drop('latest_level')
ftr_level.show(truncate=False)


# In[ ]:





# In[ ]:





# #### create feature : number of unique songs played per userId

# In[39]:


ftr_num_unique_songs = cleaned_df.filter(cleaned_df["page"] == "NextSong").select(["userId","song"]).dropDuplicates(["userId","song"]).groupby(["userId"]).count()
ftr_num_unique_songs = ftr_num_unique_songs.selectExpr("userId as userId", "count as num_unique_songs")
ftr_num_unique_songs.show(truncate=False)


# #### create feature : number of total songs played per userId

# In[40]:


ftr_num_total_songs = cleaned_df.filter(cleaned_df["page"] == "NextSong").select(["userId","song"]).groupby(["userId"]).count()
ftr_num_total_songs = ftr_num_total_songs.selectExpr("userId as userId", "count as num_total_songs")
ftr_num_total_songs.show(truncate=False)


# #### number of unique artists per userId

# In[41]:


ftr_num_unique_artists = cleaned_df.filter(cleaned_df["page"] == "NextSong").select(["userId","artist"]).dropDuplicates(["userId","artist"]).groupby(["userId"]).count()
ftr_num_unique_artists = ftr_num_unique_artists.selectExpr("userId as userId", "count as num_unique_artists")
ftr_num_unique_artists.show(truncate=False)


# #### create feature : number of Ads action per userId

# In[42]:


roll_advert_event = udf(lambda x: 1.0 if x == "Roll Advert" else 0.0, FloatType())


# In[43]:


ftr_num_ads = cleaned_df.withColumn("roll_advert_count", roll_advert_event("page"))
ftr_num_ads.show(5, truncate=False)
                                    

ftr_num_ads = ftr_num_ads.filter(ftr_num_ads["roll_advert_count"] == 1).select(["userId","roll_advert_count"]).groupby(["userId"]).count()
ftr_num_ads = ftr_num_ads.selectExpr("userId as userId", "count as roll_advert_count")
ftr_num_ads.show(truncate=False)


# #### create feature: number of thumb down action per userId

# In[44]:


thumbs_down_event = udf(lambda x: 1.0 if x == "Thumbs Down" else 0.0, FloatType())


# In[45]:


ftr_num_thumbs_down = cleaned_df.withColumn("num_thumbs_down", thumbs_down_event("page"))
ftr_num_thumbs_down.show(5, truncate=False)
                                    

ftr_num_thumbs_down = ftr_num_thumbs_down.filter(ftr_num_thumbs_down["num_thumbs_down"] == 1).select(["userId","num_thumbs_down"]).groupby(["userId"]).count()
ftr_num_thumbs_down = ftr_num_thumbs_down.selectExpr("userId as userId", "count as num_thumbs_down")
ftr_num_thumbs_down.show(truncate=False)


# #### create feature: number of thumb up action per userId

# In[46]:


thumbs_up_event = udf(lambda x: 1.0 if x == "Thumbs Up" else 0.0, FloatType())


# In[47]:


ftr_num_thumbs_up = cleaned_df.withColumn("num_thumbs_up", thumbs_up_event("page"))
ftr_num_thumbs_up.show(5, truncate=False)
                                    

ftr_num_thumbs_up = ftr_num_thumbs_up.filter(ftr_num_thumbs_up["num_thumbs_up"] == 1).select(["userId","num_thumbs_up"]).groupby(["userId"]).count()
ftr_num_thumbs_up = ftr_num_thumbs_up.selectExpr("userId as userId", "count as num_thumbs_up")
ftr_num_thumbs_up.show(truncate=False)


# #### create feature: number of friends added per userId

# In[48]:


add_friends_event = udf(lambda x: 1.0 if x == "Add Friend" else 0.0, FloatType())


# In[49]:


ftr_num_add_friends = cleaned_df.withColumn("num_add_friends", add_friends_event("page"))
ftr_num_add_friends.show(5, truncate=False)
                                    

ftr_num_add_friends = ftr_num_add_friends.filter(ftr_num_add_friends["num_add_friends"] == 1).select(["userId","num_add_friends"]).groupby(["userId"]).count()
ftr_num_add_friends = ftr_num_add_friends.selectExpr("userId as userId", "count as num_add_friends")
ftr_num_add_friends.show(truncate=False)


# ####  create featrure: number of days after initial registration per userId

# In[50]:


latest_action_df =  cleaned_df.groupBy('userId').agg(max('ts').alias('latest_action'))
print(latest_action_df.count())
num_days_after_registration_df = latest_action_df.join(cleaned_df, on='userId', how="inner").withColumn('num_days_after_registration', ((col('latest_action')-col('registration'))/86400000).cast(FloatType())).select(["userId","num_days_after_registration"]).dropDuplicates()
print(num_days_after_registration_df.count())
num_days_after_registration_df.show(5, truncate=False)


# #### create label field: churn 

# In[51]:


label_churn_df = cleaned_df.select("userId", "churn").dropDuplicates().groupby("userId", "churn").count().drop('count')
label_churn_df = label_churn_df.selectExpr("userId as userId", "churn as label")
print(label_churn_df.count())
label_churn_df.show(truncate=False)


# In[52]:


features_df = ftr_num_unique_songs                 .join(ftr_num_total_songs, ["userId"])                 .join(ftr_num_unique_artists, ["userId"])                 .join(ftr_num_ads, ["userId"])                 .join(ftr_num_thumbs_down, ["userId"])                 .join(ftr_num_thumbs_up, ["userId"])                 .join(ftr_num_add_friends, ["userId"])                 .join(num_days_after_registration_df, ["userId"])                 .join(ftr_gender, ["userId"])                 .join(ftr_level, ["userId"])                 .join(label_churn_df, ["userId"])                 

print(features_df.count())
features_df.show(truncate=False)



# 

# In[53]:


features_df =features_df.drop("userId")
features_df.show(truncate=False)


# # Modeling
# Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

# ### convert features to vectors and then scale them to standard normal before training the model

# In[54]:



assembler = VectorAssembler(inputCols = ["num_unique_songs", "num_total_songs", "num_unique_artists", "roll_advert_count", "num_thumbs_down", "num_thumbs_up", "num_add_friends", "num_days_after_registration", "gender_value", "level_value"], outputCol = "vector_features")
features_df = assembler.transform(features_df)
features_df.show(truncate=False)


# In[ ]:





# In[55]:


scaler = StandardScaler(inputCol="vector_features", outputCol="features", withStd=True)
scaler_model = scaler.fit(features_df)
features_df = scaler_model.transform(features_df)
features_df.show(truncate=False)


# In[56]:


print(features_df.count())


# ### Stratify sample the train, valid and test dataset for imbalanced churn prediction

# - Train and validation split together form 60% of the dataset (train is 60% and validation is 40%)
# - Test forms 40% of the dataset

# In[57]:


train_valid_df = features_df.stat.sampleBy("label", fractions={0: 0.6, 1: 0.4}, seed = 41)
print(f"Train and validation data count: {str(train_valid_df.count())}")
train_valid_df.groupby("label").count().show(truncate=False)


# In[68]:


train = train_valid_df.stat.sampleBy("label", fractions={0: 0.6, 1: 0.4}, seed = 41)
print(f"Train data count: {str(train.count())}")
train.groupby("label").count().show(truncate=False)


# In[69]:


valid = train_valid_df.subtract(train)
print(f"Valid data count: {str(valid.count())}")
valid.groupby("label").count().show(truncate=False)


# In[70]:


test = features_df.subtract(train_valid_df)
print(f"Test data count: {str(test.count())}")
test.groupby("label").count().show(truncate=False)


# #### Model Selection

# In[71]:


lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=50)
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', seed=9)
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', seed=9)
gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxIter=10, seed=9)


# In[72]:


models = [lr,dt,rf,gbt]


# In[73]:


evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol='prediction')


# Use the train and validation data to identify which model performs well amongst the ones which are chosen

# In[74]:


get_ipython().run_cell_magic('time', '', 'for model in models:\n    print(f\'Start Training >>>> {type(model)}\')\n    model = model.fit(train)\n    \n    print(f\'Start Prediction >>>> {type(model)}\')\n    predictions = model.transform(valid)\n    \n    print(f\'F1 for {type(model)} is: {evaluator.evaluate(predictions, {evaluator.metricName: "f1"})}\')\n    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / (predictions.count())\n    print(f"{type(model)} Accuracy = {accuracy}")\n    print("-------------------------------------------------------------")')


# #### Selected Model Tuning

# In[75]:


paramsGrid = ParamGridBuilder()     .addGrid(rf.numTrees,[25, 50])     .addGrid(rf.maxDepth,[5, 10])     .build()


crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramsGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName = "f1"),
                          numFolds=5)


# In[76]:


get_ipython().run_cell_magic('time', '', 'cvModel = crossval.fit(train)')


# In[77]:


cvModel.avgMetrics


# #### Used the tuned-model for final prediction on test data

# In[78]:


results = cvModel.transform(test)


# In[79]:


accuracy = results.filter(results.label == results.prediction).count() / (results.count())
print(accuracy)


# In[80]:


print(f'F1 for {type(cvModel)} is: {evaluator.evaluate(results, {evaluator.metricName: "f1"})}')


# In[82]:


print(f'Recall for {type(cvModel)} is: {evaluator.evaluate(results, {evaluator.metricName: "weightedRecall"})}')


# In[83]:


print(f'Precision for {type(cvModel)} is: {evaluator.evaluate(results, {evaluator.metricName: "weightedPrecision"})}')


# # Final Steps
# Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




