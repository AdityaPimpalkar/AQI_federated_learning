import pandas as pd
import math
import numpy as np
import tensorflow as tf

NUM_CLIENTS = 20
cid=1
dublin_port = pd.read_csv('/Users/adityapimpalkar/Documents/AQI_new/AQI_federated_learning/data/dublin_port/dublin_port.csv')
dublin_port = dublin_port.drop(['datetime' ,'Unnamed: 0'], axis=1)

features = dublin_port.loc[:, ~dublin_port.columns.isin(['pm10', 'pm2.5'])]
target = dublin_port.loc[:, dublin_port.columns.isin(['pm10', 'pm2.5'])]

partition_size = math.floor(len(features) / NUM_CLIENTS)
idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
x_train_cid = features[idx_from:idx_to]
y_train_cid = target[idx_from:idx_to]

dataset = tf.keras.utils.timeseries_dataset_from_array(dublin_port.to_numpy(), targets=['pm10', 'pm2.5'], sequence_length=1, batch_size=1, sequence_stride=1, shuffle=False)

total_samples = len(dublin_port)
train_samples = int(0.7 * total_samples)
val_samples = int(0.15 * total_samples)
test_size = total_samples - train_samples - val_samples

train_dataset = dataset.take(train_samples)
remaining_dataset = dataset.skip(train_samples)

val_dataset = remaining_dataset.take(val_samples)
test_dataset = remaining_dataset.skip(val_samples)


for batch in train_dataset.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

print(inputs.shape[1], inputs.shape[2])

#partition_size = len(dublin_port) // NUM_CLIENTS
#lengths = [partition_size] * NUM_CLIENTS
#print(np.array_split(dublin_port, 3))



