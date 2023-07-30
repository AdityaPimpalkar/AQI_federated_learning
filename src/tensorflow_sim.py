import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import collections
import time
import math
import os

data = pd.read_csv(os.getcwd() + '/data/dublin_port/dublin_port.csv')
#dropping uncessary columns
data = data.drop(['datetime' ,'Unnamed: 0'], axis=1)

#lengths for train test and validation sets 70-20-20
total_samples = len(data)
train_samples = int(0.7 * total_samples)
val_samples = int(0.15 * total_samples)

#subsetting the original data as per set lengths
train_data = data[:train_samples]
val_data = data[train_samples:train_samples + val_samples]
test_data = data[train_samples + val_samples:]

#setting number of clients
NUM_CLIENTS = 20

def create_clients(data, clients):
    '''
    data - pandas dataframe
    clients - number of FL clients to split the data
    '''
    datasets = []
    for cid in range(clients):
        #creating a partition value as per the number of FL clients
        partition_size = math.floor(len(data) / NUM_CLIENTS)
        #creating indexes to subset the dataframe
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
        #subsetting dataframe
        client_batch_data = data[idx_from:idx_to]
        features = client_batch_data.loc[:, ~client_batch_data.columns.isin(["pm2.5", "pm10"])]
        targets = client_batch_data.loc[:, client_batch_data.columns.isin(["pm2.5", "pm10"])]
        #creating a timeseries datase
        dataset = tf.keras.utils.timeseries_dataset_from_array(features.to_numpy(), 
                                                                targets.to_numpy(), 
                                                                #interval period, we have data at every 1 hour intervals
                                                                sequence_length=1,
                                                                batch_size=1, 
                                                                sequence_stride=1, 
                                                                shuffle=False)
        datasets.append(dataset)
    return datasets

def model_fn():
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=(None, None, 9), dtype=tf.float64, name=None),
        y=tf.TensorSpec(shape=(None, 2), dtype=tf.float64, name=None))
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1, 9)),
            tf.keras.layers.Conv1D(filters=60, kernel_size=5,
              strides=1, padding="causal",
              activation="relu"),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    return tff.learning.models.from_keras_model(
      model,
      input_spec = input_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()])


trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02))

def train(num_rounds=10):
    state = trainer.initialize()
    datasets = create_clients(train_data, NUM_CLIENTS)
    for _ in range(num_rounds):
        t1 = time.time()
        result = trainer.next(state, datasets)
        state = result.state
        train_metrics = result.metrics['client_work']['train']
        t2 = time.time()
        print('train metrics {m}, round time {t:.2f} seconds'.format(
            m=train_metrics, t=t2 - t1))



train()