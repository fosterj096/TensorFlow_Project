# Justin Foster
# CS461 - Program 3
# Tensorflow Twitter Analysis

# Using Tensorflow was very new for me so I will list all resources here:
# https://www.youtube.com/watch?v=tPYj3fFJGjk || tf tutorial
# https://www.youtube.com/watch?v=6_2hzRopPbQ || tf + python tutorial
# https://www.tensorflow.org/api_docs/python/tf/all_symbols || tf documentation
# https://www.youtube.com/watch?v=Jmrn8ukp8b8 || tf tutorial
# https://www.tensorflow.org/tutorials/load_data/csv || tf with csv's
#
#

# Importing depedencies
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #standard neurel network, uses "relu"
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score

# Pandas line that reads/formats csv files
twitter = pd.read_csv('tweet.csv', names = [ 'favorite_count','full_text','hashtags', 'retweet_count',
                      'year','party_id'])
twitter.head()
twitter_features = twitter.copy()


inputs = {}

# Formatting required for working with different data types
for name, column in twitter_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape = (1, ), name = name, dtype = dtype)


