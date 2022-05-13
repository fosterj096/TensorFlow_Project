# Justin Foster
# CS461 - Program 3
# Tensorflow Twitter Analysis

# Using Tensorflow was very new for me so I will list all resources here:
# https://www.youtube.com/watch?v=tPYj3fFJGjk || tf tutorial
# https://www.youtube.com/watch?v=6_2hzRopPbQ || tf + python tutorial
# https://www.tensorflow.org/api_docs/python/tf/all_symbols || tf documentation
# https://www.youtube.com/watch?v=Jmrn8ukp8b8 || tf tutorial
# https://www.tensorflow.org/tutorials/load_data/csv#mixed_data_types || tf with csv's
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
# https://stackoverflow.com/questions/62436302/extract-target-from-tensorflow-prefetchdataset
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://keras.io/api/layers/activations/

# Importing depedencies
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.models import *
from sklearn.metrics import accuracy_score
np.set_printoptions(precision = 3, suppress = True)

# Pandas line that reads/formats csv files
twitter = pd.read_csv('tweet.csv')
twitter.head()
# Ignoring the full_text as mentioned in the assignment description
twitter.drop(columns = 'full_text')
# Setting party_id as our target variable for prediction, converting R/D into binary values
twitter['party_id'] = np.where(twitter['party_id'] == 'R', 0, 1)

# Making sets for training, validation, and testing
train, validate, test = np.split(twitter.sample(frac = 1), [int(.8*len(twitter)), int(.9*len(twitter))])

# Setting twitter features equal to our training split because.. it is the training split
twitter_features = train.copy()
# Pop party_id as its the value we are trying to predict then converting to a numpy array due to errors
twitter_labels = twitter_features.pop('party_id')
twitter_labels = twitter_labels.to_numpy()

# Formatting required for working with different data types, these are for the
# "symbolic" aka non-numeric values - they are all converted to tf float32 type
inputs = {}
for name, column in twitter_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape = (1, ), name = name, dtype = dtype)


'''
labelInputs = {}
for name, column in twitter_labels.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    labelInputs[name] = tf.keras.Input(shape = (1, ), name = name, dtype = dtype)
'''
# Combining all layers of any sort of numeric data and combining them for
# normalization
integerInput = {name:input for name, input in inputs.items()
                  if input.dtype == tf.float32}

x = layers.Concatenate()(list(integerInput.values()))
norm = layers.Normalization()
norm.adapt(np.array(twitter_features[integerInput.keys()]))
allInputs = norm(x)
rawSymbolicInputs = [allInputs]

# More conversion to float32 type
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    # Functions that map strings to tf's integer indices which are then mapped to the value
    # of float32 that is used by the model
    lookup = layers.StringLookup(vocabulary = np.unique(twitter_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens = lookup.vocabulary_size())
    
    # Running above functions to create the vector for the model
    x = lookup(input)
    x = one_hot(x)
    rawSymbolicInputs.append(x)

# Concatening all of the data that was previously converted to float32
# tf.keras.utils.plot_model() is used to convert to a dot model
# rankDir is a python compatible formatting tool, dpi = dots per inch,
rawInputsConcatenate = layers.Concatenate()(rawSymbolicInputs)
twitter_processing = tf.keras.Model(inputs, rawInputsConcatenate)

# Converting pandas dataframe into a dictionary of tensor so tensorflow can work 
twitter_features_dictionary = {name: np.array(value)
                               for name, value in twitter_features.items()}

featureDictionary = {name: values[:1] for name, values in twitter_features_dictionary.items()}
twitter_processing(featureDictionary)

# Building tensorflow model
def twitterModel(preprocessing_head, inputs):
    # I think "relu" is the correct activation, documentation states its linear
    # softmax is a probability distribution?
    # https://keras.io/api/layers/activations/
    body = tf.keras.Sequential([layers.Dense(64, activation = 'relu'), layers.Dense(1)])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    # https://keras.io/api/metrics/accuracy_metrics/
    model.compile(loss = tf.losses.BinaryCrossentropy(from_logits = True),
                  optimizer = tf.optimizers.Adam(), metrics = ['Accuracy'])

    return model

twitterModel = twitterModel(twitter_processing, inputs)
# Training the model with our twitter dictionary and the seperate model "twitter_labels"
# which will be used to predict the party_id
# https://www.tensorflow.org/guide/keras/train_and_evaluate
twitterModel.fit(x = twitter_features_dictionary, y = twitter_labels, epochs = 1, verbose = 1)
features_dictionary = {name:values[:1] for name, values in twitter_features_dictionary.items()}
