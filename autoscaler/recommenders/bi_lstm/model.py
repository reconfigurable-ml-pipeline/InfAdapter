import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from settings import BASE_DIR


def get_x_y(data):
    x = []
    y = []

    for i in range(len(data) - 10):
        x.append(data[i:i+10])
        y.append(data[i+10])
    return x, y


def get_data():
    with open(f"{BASE_DIR}/autoscaler/dataset/worldcup/workload.txt", "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))
    workload_minute = []
    for i in range(0, len(workload), 60):
        workload_minute.append(sum(workload[i:i+60]))

    fragment_index = -3*len(workload_minute)//10
    train_data = workload_minute[:fragment_index]
    test_data = workload_minute[fragment_index:]

    train_x, train_y = get_x_y(train_data)
    test_x, test_y = get_x_y(test_data)

    return (
        tf.convert_to_tensor(np.array(train_x).reshape((-1, 10, 1)), dtype=tf.float32),
        tf.convert_to_tensor(np.array(train_y), dtype=tf.float32),
        tf.convert_to_tensor(np.array(test_x).reshape((-1, 10, 1)), dtype=tf.float32),
        tf.convert_to_tensor(np.array(test_y), dtype=tf.float32)
    )


def create_model():
    model = Sequential()
    model.add(Input(shape=(10, 1)))
    model.add(Bidirectional(LSTM(30, activation="relu")))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_data()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    model = create_model()
    # print(model.summary())
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=50, batch_size=64, validation_data=(test_x, test_y))
    predictions = model.predict(test_x)
    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(predictions), label="predictions")
    plt.legend()
    plt.show()
    model.save(f"{BASE_DIR}/autoscaler/recommenders/bi_lstm/saved")
