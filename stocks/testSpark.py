#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:39:16 2018

@author: uwe.mueller
"""
import findspark
import pyspark


findspark.init()
sc = pyspark.SparkContext(master="spark://192.168.243.136:7077", appName="mySpyderApp")



