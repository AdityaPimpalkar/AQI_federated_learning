import os
import math

# Make TensorFlow logs less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        #split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        split_idx = math.floor(len(x_train) * 0.7)  # Use 30% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        #self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        #self.model.fit(self.x_train, self.y_train, epochs=15, verbose=1)
        self.model.fit(self.x_train, self.y_train, epochs=50, verbose=1)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile("sgd", "sparse_categorical_crossentropy", metrics=["accuracy"])

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / 50)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    return FlwrClient(model, x_train_cid, y_train_cid)


def main() -> None:
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=50,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.2,
            fraction_evaluate=0.4,
            min_fit_clients=50,
            min_evaluate_clients=30,
            min_available_clients=50,
        ),
    )


if __name__ == "__main__":
    main()
