from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql.functions import monotonically_increasing_id 

if __name__ == "__main__":
    # The main script - create our SparkContext
    conf = SparkConf().setAppName("SparkML_kmeans")
    sc = SparkContext(conf = conf)

    
    path = 'hdfs://hadoop.arat.dbmc.comp.db.de:8020/BR401_traction/trainingdata_072.parquet'
    sqlContext = SQLContext(sc)
    FEATURES_COL = ['Min_01', 'Max_01', 'RMS_01']

    df = sqlContext.read.parquet(path) # requires spark 2.0
    df = df.select("*").withColumn("id", monotonically_increasing_id())    

    for col in df.columns:
        if col in FEATURES_COL:
            df = df.withColumn(col,df[col].cast('float'))
    print(type(df))


    dfraw = df.toPandas().set_index('id')
    #fil = dfraw['RMS_01'] >= 0.75
    #dfraw = dfraw[fil]
    #dfraw['T'] = pd.Timestamp(dfraw['time']).tz_localize('UTC')    
    print(dfraw.head())
 
    #EFfil = ((dfraw['time'] >= 1558385194000) & (dfraw['time'] <= 1558385197000))
    #EF = dfraw[EFfil]    
    #print(EF.head())

    raw = plt.figure(figsize=(12,10)).gca(projection='3d')
    raw.scatter(dfraw.Min_01, dfraw.Max_01, dfraw.RMS_01)
    raw.set_xlabel('Min_01')
    raw.set_ylabel('Max_01')
    raw.set_zlabel('RMS_01')
    plt.savefig('rawdataSR1_training')
    
    vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
    df_kmeans = vecAssembler.transform(df).select('id', 'features')
    
    cost = np.zeros(20)
    for k in range(2,20):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
        cost[k] = model.computeCost(df_kmeans) # requires Spark 2.0 or later

    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(2,20),cost[2:20]) 
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    ax.set_title('Costfunction k-means modelling')
    fig.savefig('costfunction.png')
    plt.show()
    
    
    k = 12
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans)
    model.save('kmeans')
    centers = model.clusterCenters()

    print("Cluster Centers: ")
    for center in centers:
        print(center)
    
    transformed = model.transform(df_kmeans).select('id', 'prediction')
    rows = transformed.collect()
    print(rows[:3])

    df_pred = sqlContext.createDataFrame(rows)
    df_pred.show()

    df_pred = df_pred.join(df, 'id')
    df_pred.show()

    pddf_pred = df_pred.toPandas().set_index('id')
    pddf_pred.head()

    colors = cm.rainbow(np.linspace(0, 1, k))
    threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
    threedee.scatter(pddf_pred.Min_01, pddf_pred.Max_01, pddf_pred.RMS_01, c=colors[pddf_pred.prediction])
    threedee.set_xlabel('Min_01')
    threedee.set_ylabel('Max_01')
    threedee.set_zlabel('RMS_01')
    threedee.set_title('k-means clustering ICE1 Stromrichter')
    plt.savefig('result_clustering_trainingdata_k12.png')
    plt.show()
    
    sc.stop()
    
