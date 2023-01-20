import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from auto_tuner import AUTO_TUNER_DIRECTORY
from lstm_forecaster import get_x_y

def min_conv(minutes):
    if minutes == 0:
        return 0
    if minutes < 60:
        return f"{minutes}m"
    if minutes % 60 == 0:
        return f"{minutes//60}h"
    return f"{minutes//60}h{minutes%60}m"

if __name__ == "__main__":
    model = load_model(f"{AUTO_TUNER_DIRECTORY}/adapter/forecasters/lstm_saved_model")
    with open(f"{AUTO_TUNER_DIRECTORY}/dataset/twitter_trace/workload.txt", "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))
    workload = list(filter(lambda x:x!=0, workload))
    
    hour = 60 * 60
    day = hour * 24
    test_idx = 18 * day
    
    test_data = workload[test_idx:test_idx + 2 * hour]
   
    test_x, test_y = get_x_y(test_data)
    
    test_x = tf.convert_to_tensor(np.array(test_x).reshape((-1, 10, 1)), dtype=tf.float32)
    
    prediction = model.predict(test_x)
    
    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(prediction), label="predictions")
    plt.xlabel("time (minute)")
    plt.ylabel("load (RPS)")
    plt.legend()
    plt.show()
    