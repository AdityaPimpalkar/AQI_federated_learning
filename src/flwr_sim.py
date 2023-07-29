import os
import math
import pandas as pd

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

NUM_CLIENTS = 20
data = pd.read_csv('/Users/adityapimpalkar/Documents/AQI_new/AQI_federated_learning/data/dublin_port/dublin_port.csv')
data = data.drop(['datetime' ,'Unnamed: 0'], axis=1)

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, client_batch_data, targets) -> None:
        super().__init__()
        self.model = model
        dataset = tf.keras.utils.timeseries_dataset_from_array(client_batch_data.to_numpy(), 
                                                                targets=targets, 
                                                                sequence_length=1, 
                                                                batch_size=1, 
                                                                sequence_stride=1, 
                                                                shuffle=False)
        total_samples = len(data)
        self.train_samples = int(0.7 * total_samples)
        self.val_samples = int(0.15 * total_samples)

        self.train_dataset = dataset.take(self.train_samples)
        remaining_dataset = dataset.skip(self.train_samples)

        self.val_dataset = remaining_dataset.take(self.val_samples)
        self.test_dataset = remaining_dataset.skip(self.val_samples)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        #self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        #return self.model.get_weights(), len(self.x_train), {}
        self.model.fit(self.train_dataset, validation_data=self.val_dataset, epochs=2, verbose=2)
        return self.model.get_weights(), self.train_samples, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.test_dataset, verbose=2)
        return loss, self.val_samples, {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1, 11)),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    model.compile("adam", metrics=["accuracy"])

    partition_size = math.floor(len(data) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    client_batch_data = data[idx_from:idx_to]

    # Create and return client
    return FlwrClient(model, client_batch_data, targets=['pm10', 'pm2.5'])


def main() -> None:
    # Start Flower simulation
    sim_result = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=NUM_CLIENTS
        ),
        #ray_init_args = {"include_dashboard": True}
    )

    print(sim_result)


if __name__ == "__main__":
    main()