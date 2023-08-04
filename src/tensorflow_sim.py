import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import keras
import keras.backend as K
import collections
import time
import math
import os

def create_dataset(data):
    features = data.loc[:, ~data.columns.isin(["pm2.5", "pm10"])]
    targets = data.loc[:, data.columns.isin(["pm2.5", "pm10"])]
    #creating a timeseries datase
    return tf.keras.utils.timeseries_dataset_from_array(
        features.to_numpy(),
        targets.to_numpy(),
        #interval period, we have data at every 1 hour intervals
        sequence_length=1,
        batch_size=1,
        sequence_stride=1,
        shuffle=False
    )

#ref - https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
class Attention(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_fl_clients(data, no_of_clients):
    '''
    data - pandas dataframe
    clients - number of FL clients to split the data
    '''
    datasets = []
    for cid in range(no_of_clients):
        #creating a partition value as per the number of FL clients
        partition_size = math.floor(len(data) / no_of_clients)
        #creating indexes to subset the dataframe
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
        #subsetting dataframe
        client_batch_data = data[idx_from:idx_to]
        dataset = create_dataset(client_batch_data)
        datasets.append(dataset)
    return datasets

def model_fn():
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=(None, None, 9), dtype=tf.float64, name=None),
        y=tf.TensorSpec(shape=(None, 2), dtype=tf.float64, name=None)
    )
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1, 9)),
            tf.keras.layers.Conv1D(
                filters=20,
                kernel_size=5,
                strides=1,
                padding="causal",
              activation="relu"
            ),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.LSTM(20, return_sequences=True),
            Attention(),
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

def train(trainer ,train_data, num_rounds, no_of_clients):
    state = trainer.initialize()
    datasets = create_fl_clients(train_data, no_of_clients)
    for _ in range(num_rounds):
        t1 = time.time()
        result = trainer.next(state, datasets)
        state = result.state
        train_metrics = result.metrics['client_work']['train']
        t2 = time.time()
        print('train metrics {m}, round time {t:.2f} seconds'.format(
            m=train_metrics, t=t2 - t1))

    return state



if __name__ == "__main__":
    #setting number of clients
    NUM_CLIENTS = 20
    ROUNDS = 8
    SERVER_LR = 0.03
    CLIENT_LR = 0.06

    data = pd.read_csv('dublin_port.csv')
    #dropping uncessary columns
    data = data.drop(['datetime' ,'Unnamed: 0'], axis=1)

    #lengths for train test and validation sets 70-15-15
    total_samples = len(data)
    train_samples = int(0.7 * total_samples)
    val_samples = int(0.15 * total_samples)

    #subsetting the original data as per set lengths
    train_data = data[:train_samples]
    val_data = data[train_samples:train_samples + val_samples]
    test_data = data[train_samples + val_samples:]

    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        #server_optimizer_fn=lambda: tf.keras.optimizers.SGD(SERVER_LR),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(CLIENT_LR)
    )

    state = train(
        trainer,
        train_data,
        num_rounds=ROUNDS,
        no_of_clients=NUM_CLIENTS
    )

    eval = tff.learning.build_federated_evaluation(model_fn)

    eval_client_data = create_dataset(val_data)
    metrics = eval(trainer.get_model_weights(state), [eval_client_data])
    print("Eval set metrics" ,metrics)

    test_client_data = create_dataset(test_data)
    metrics = eval(trainer.get_model_weights(state), [test_client_data])
    print("Test set metrics" ,metrics)

