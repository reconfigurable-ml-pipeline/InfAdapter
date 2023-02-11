import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import regularizers


from auto_tuner import AUTO_TUNER_DIRECTORY


def get_x_y(data):
    x = []
    y = []
    history_seconds = 600
    for i in range(0, len(data) - history_seconds, 60):
        t = data[i:i+history_seconds]
        for j in range(0, len(t), 60):
            x.append(max(t[j:j+60]))
        y.append(max(data[i+history_seconds:i+history_seconds+60]))
    return x, y


def get_data():
    with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))

    workload = list(filter(lambda x:x!=0, workload))
    train_to_idx = 14 * 24 * 60 * 60
    workload_train = workload[:train_to_idx]
    workload_test = workload[train_to_idx:]

    train_x, train_y = get_x_y(workload_train)
    test_x, test_y = get_x_y(workload_test)

    return (
        tf.convert_to_tensor(np.array(train_x).reshape((-1, 10, 1)), dtype=tf.int32),
        tf.convert_to_tensor(np.array(train_y), dtype=tf.int32),
        tf.convert_to_tensor(np.array(test_x).reshape((-1, 10, 1)), dtype=tf.int32),
        tf.convert_to_tensor(np.array(test_y), dtype=tf.int32)
    )


def create_model():
    model = Sequential()
    model.add(Input(shape=(10, 1)))
    model.add(LSTM(25, activation="relu", kernel_regularizer=regularizers.L1(0.00001)))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    tf.random.set_seed(7)
    train_x, train_y, test_x, test_y = get_data()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    model = create_model()
    print(model.summary())
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y))
    predictions = model.predict(test_x)
    
    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(predictions), label="predictions")
    plt.legend()
    plt.show()
    model.save(f"{AUTO_TUNER_DIRECTORY}/adapter/forecasters/lstm_saved_model")
