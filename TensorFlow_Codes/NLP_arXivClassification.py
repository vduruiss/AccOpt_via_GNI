# Large-Scale Multi-Label Text Classification

'''
We test the eBrAVO and pBrAVO algorithms for
Large-Scale Multi-Label Text Classification

We consider the Natural Language Processing problem of constructing
a multi-label text classifier which can provide suggestions for the
most appropriate subject areas for arXiv papers based on their abstracts.

More details can be found in
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
     Authors: Valentin Duruisseaux and Melvin Leok. 

Usage:

	python ./TensorFlow_Codes/NLP_arXivClassification.py




Based on keras.io/examples/nlp/multi_label_classification/

Authors of the original code:
[Sayak Paul](https://twitter.com/RisingSayak)
[Soumik Rakshit](https://github.com/soumik12345)

'''

################################################################################

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import BrAVO_tf


################################################################################
# Data Loading and Preparation

# Load Dataset
arxiv_data = pd.read_csv(
    "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
)
arxiv_data.head()

# Remove Duplicates
total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]

# Filtering the rare terms.
arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
arxiv_data_filtered.shape

# Convert the string labels to lists of strings
arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(lambda x: literal_eval(x))
arxiv_data_filtered["terms"].values[:5]

# Initial train and test split.
test_split = 0.15
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["terms"].values,
    )

# Splitting the test set further into validation and new test sets.
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

# Process Labels using the 'StringLookup' layer
terms = tf.ragged.constant(train_df["terms"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()

# Separating the individual unique classes available from the label pool and
# then using this information to represent a given label set with 0's and 1's.
sample_label = train_df["terms"].iloc[0]
label_binarized = lookup([sample_label])

train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()

max_seqlen = 150
batch_size = 64
padding_token = "<pad>"
auto = tf.data.AUTOTUNE

def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)

# Prepare the `tf.data.Dataset` objects
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

# Vectorization
vocabulary = set()
train_df["summaries"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)

text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
    ).prefetch(auto)


################################################################################
# Text Classification Model

def make_model():
    model = keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid")]
    )
    return model


################################################################################
## Train the models

epochs = 20

###########################
# With ADAM
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)

model1 = make_model()

model1.compile(loss="binary_crossentropy", optimizer=optimizer1)
print("\n\n --------------- \n ADAM Training \n --------------- \n")
history1 = model1.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


###########################
# With SGD
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.2)

model2 = make_model()
model2.compile(loss="binary_crossentropy", optimizer=optimizer2)
print("\n\n ---------------  \n SGD Training \n ---------------  \n")
history2 = model2.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)

###########################
# With eBrAVO
optimizer3 = BrAVO_tf.eBravo(learning_rate=0.02, C = 1e4)

model3 = make_model()
model3.compile(loss="binary_crossentropy", optimizer=optimizer3)
print("\n\n ---------------  \n eBrAVO Training \n ---------------  \n")

history3 = model3.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)

###########################
# With pBrAVO
optimizer4 = BrAVO_tf.pBravo(learning_rate=0.05, C = 0.1)

model4 = make_model()
model4.compile(loss="binary_crossentropy", optimizer=optimizer4)
print("\n\n ---------------  \n pBrAVO Training \n ---------------  \n")
history4 = model4.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


################################################################################
# Plotting

plt.subplots(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(np.arange(1,epochs+1),history1.history["loss"], 'black', linewidth = 2, label="ADAM")
plt.plot(np.arange(1,epochs+1),history2.history["loss"], 'green', linewidth = 2, label="SGD")
plt.plot(np.arange(1,epochs+1),history3.history["loss"], 'blue', linewidth = 2, label="eBrAVO")
plt.plot(np.arange(1,epochs+1),history4.history["loss"], 'red', linewidth = 2, label="pBrAVO")
plt.xlabel("epochs",fontsize=14)
plt.ylabel("Training Loss",fontsize=14)
plt.yscale("log")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(1,epochs+1),history1.history["val_loss"], 'black', linewidth = 2, label="ADAM")
plt.plot(np.arange(1,epochs+1),history2.history["val_loss"], 'green', linewidth = 2, label="SGD")
plt.plot(np.arange(1,epochs+1),history3.history["val_loss"], 'blue', linewidth = 2, label="eBrAVO")
plt.plot(np.arange(1,epochs+1),history4.history["val_loss"], 'red', linewidth = 2, label="pBrAVO")
plt.xlabel("epochs",fontsize=14)
plt.ylabel("Validation Loss",fontsize=14)
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()
