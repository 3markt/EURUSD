#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
import pyspark

basepath = '/Users/uwe.muller/Hope/'
path = basepath + 'data/nifi/'

a = np.random.random(100)
b = np.random.randint(1,100,100)
d = np.stack((b,a), axis=1)

df = pd.DataFrame(d, columns=['Int', 'Rand'])

spark = pyspark.sql.SparkSession.builder.appName("LCU-NiFi-Test").getOrCreate()
sdf = spark.createDataFrame(df)
sdf.write.parquet(path + "pTest", mode = "append")


