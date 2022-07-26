# Timeseries Forecasting for Weather Prediction

"""
We test the eBrAVO and pBrAVO algorithms for timeseries forecasting for weather prediction

We use a Long Short-Term Memory (LSTM) model (with 5153 parameters).

More details can be found in
    "Practical Perspectives on Symplectic Accelerated Optimization"
    Authors: Valentin Duruisseaux and Melvin Leok. 2022.

Usage:

	python ./TensorFlow_Codes/WeatherForecasting.py



Based on keras.io/examples/timeseries/timeseries_weather_forecasting/

Authors of the original code:
[Prabhanshu Attri](https://prabhanshu.com/github)
[Yashika Sharma](https://github.com/yashika51)
[Kristi Takach](https://github.com/ktakattack)
[Falak Shah](https://github.com/falaktheoptimist)

We will be using Jena Climate dataset recorded by the
[Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/).
The dataset consists of 14 features such as temperature, pressure, humidity etc,
recorded once per 10 minutes.

The table below shows the column names, their value formats, and their description.

Index| Features      |Format             |Description
-----|---------------|-------------------|-----------------------
1    |Date Time      |01.01.2009 00:10:00|Date-time reference
2    |p (mbar)       |996.52             |The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.
3    |T (degC)       |-8.02              |Temperature in Celsius
4    |Tpot (K)       |265.4              |Temperature in Kelvin
5    |Tdew (degC)    |-8.9               |Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.
6    |rh (%)         |93.3               |Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.
7    |VPmax (mbar)   |3.33               |Saturation vapor pressure
8    |VPact (mbar)   |3.11               |Vapor pressure
9    |VPdef (mbar)   |0.22               |Vapor pressure deficit
10   |sh (g/kg)      |1.94               |Specific humidity
11   |H2OC (mmol/mol)|3.12               |Water vapor concentration
12   |rho (g/m ** 3) |1307.75            |Airtight
13   |wv (m/s)       |1.03               |Wind speed
14   |max. wv (m/s)  |1.75               |Maximum wind speed
15   |wd (deg)       |152.3              |Wind direction in degrees
"""


################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile
import BrAVO_tf

################################################################################
# Data Preparation

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"


"""
Here we are picking ~300,000 data points for training. Observation is recorded every
10 mins, that means 6 times per hour. We will resample one point per hour since no
drastic change is expected within 60 minutes. We do this via the `sampling_rate`
argument in `timeseries_dataset_from_array` utility.

We are tracking data from past 720 timestamps (720/6=120 hours). This data will be
used to predict the temperature after 72 timestamps (72/6=12 hours).

Since every feature has values with varying ranges, we do normalization to
confine feature values to a range of `[0, 1]` before training a neural network.
We do this by subtracting the mean and dividing by the standard deviation of each feature.

71.5 % of the data will be used to train the model, i.e. 300,693 rows.
`split_fraction` can be changed to alter this percentage.

The model is shown data for first 5 days i.e. 720 observations, that are sampled
every hour. The temperature after 72 (12 hours * 6 observation per hour)
observation will be used as a label.
"""

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
features = df[selected_features]
features.index = df[date_time_key]
features = normalize(features.values, train_split)
features = pd.DataFrame(features)

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

past = 720
future = 72

batch_size = 256


############################
# Training dataset
"""
The training dataset labels starts from the 792nd observation (720 + 72).
"""

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(7)]].values
y_train = features.iloc[start:end][[1]]

sequence_length = int(past / step)

"""
The `timeseries_dataset_from_array` function takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as length of the
sequences/windows, spacing between two sequence/windows, etc., to produce batches of
sub-timeseries inputs and targets sampled from the main timeseries.
"""

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

############################
# Validation dataset
"""
The validation dataset must not contain the last 792 rows as we won't have label data for
those records, hence 792 must be subtracted from the end of the data.

The validation label dataset must start from 792 after train_split, hence we must add
past + future (792) to label_start.
"""

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
y_val = features.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


################################################################################
# Training

epochs = 20

###########################
# With ADAM
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)

for batch in dataset_train.take(1):
    inputs1, targets1 = batch

inputs1 = keras.layers.Input(shape=(inputs1.shape[1], inputs1.shape[2]))
lstm_out1 = keras.layers.LSTM(32)(inputs1)
outputs1 = keras.layers.Dense(1)(lstm_out1)

model1 = keras.Model(inputs=inputs1, outputs=outputs1)

model1.compile(loss="mse", optimizer=optimizer1)

print("\n\n --------------- \n ADAM Training \n --------------- \n")
history1 = model1.fit(dataset_train, epochs=epochs,validation_data=dataset_val)

###########################
# With SGD
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.01)

for batch in dataset_train.take(1):
    inputs2, targets2 = batch

inputs2 = keras.layers.Input(shape=(inputs2.shape[1], inputs2.shape[2]))
lstm_out2 = keras.layers.LSTM(32)(inputs2)
outputs2 = keras.layers.Dense(1)(lstm_out2)

model2 = keras.Model(inputs=inputs2, outputs=outputs2)

model2.compile(loss="mse", optimizer=optimizer2)
print("\n\n --------------- \n SGD Training \n --------------- \n")
history2 = model2.fit(dataset_train, epochs=epochs,validation_data=dataset_val)

###########################
# With eBrAVO
optimizer3 = BrAVO_tf.eBravo(learning_rate=12)

for batch in dataset_train.take(1):
    inputs3, targets3 = batch

inputs3 = keras.layers.Input(shape=(inputs3.shape[1], inputs3.shape[2]))
lstm_out3 = keras.layers.LSTM(32)(inputs3)
outputs3 = keras.layers.Dense(1)(lstm_out3)

model3 = keras.Model(inputs=inputs3, outputs=outputs3)

model3.compile(loss="mse", optimizer=optimizer3)
print("\n\n --------------- \n eBrAVO Training \n --------------- \n")
history3 = model3.fit(dataset_train, epochs=epochs,validation_data=dataset_val)

###########################
# With pBrAVO
optimizer4 = BrAVO_tf.pBravo(learning_rate=0.05)

for batch in dataset_train.take(1):
    inputs4, targets4 = batch

inputs4 = keras.layers.Input(shape=(inputs4.shape[1], inputs4.shape[2]))
lstm_out4 = keras.layers.LSTM(32)(inputs4)
outputs4 = keras.layers.Dense(1)(lstm_out4)

model4 = keras.Model(inputs=inputs4, outputs=outputs4)

model4.compile(loss="mse", optimizer=optimizer4)
print("\n\n --------------- \n pBrAVO Training \n --------------- \n")
history4 = model4.fit(dataset_train, epochs=epochs,validation_data=dataset_val)


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
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(1,epochs+1),history1.history["val_loss"], 'black', linewidth = 2, label="ADAM")
plt.plot(np.arange(1,epochs+1),history2.history["val_loss"], 'green', linewidth = 2, label="SGD")
plt.plot(np.arange(1,epochs+1),history3.history["val_loss"], 'blue', linewidth = 2, label="eBrAVO")
plt.plot(np.arange(1,epochs+1),history4.history["val_loss"], 'red', linewidth = 2, label="pBrAVO")
plt.xlabel("epochs",fontsize=14)
plt.ylabel("Validation Loss",fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()
