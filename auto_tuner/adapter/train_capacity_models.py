import os
from joblib import dump
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from auto_tuner import AUTO_TUNER_DIRECTORY

os.system(f"mkdir -p {AUTO_TUNER_DIRECTORY}/adapter/capacity_models")

df = pd.read_csv(f"{AUTO_TUNER_DIRECTORY}/adapter/capacity_result_Jan_11.csv")
df = df[df["SLA"] == 750]

versions = [18, 34, 50, 101, 152]
models = {}
for version in versions:
    X_train = np.array(df[(df["CPU"] <= 10) & (df["ARCH"] == version)]["CPU"]).reshape(-1, 1)
    Y_train = df[(df["CPU"] <= 10) & (df["ARCH"] == version)]["capacity"]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    dump(model, f"{AUTO_TUNER_DIRECTORY}/adapter/capacity_models/resnet-{version}.joblib")
    models[version] = model

for version in versions:
    model = models[version]
    X_test = np.array(df[df["ARCH"] == version]["CPU"]).reshape(-1, 1)

    y_pred = model.predict(X_test)
    plt.title(f"capacity prediction for resnet{version}")
    plt.plot(list(range(1, 21)), y_pred, "bo", label="prediction")
    plt.plot(list(range(1, 21)), df[df["ARCH"] == version]["capacity"], "go", label="profiled")
    plt.xlabel("CPU cores")
    plt.ylabel("capacity")
    plt.xticks(list(range(1, 21)))
    plt.legend()
    plt.show()
    plt.close()
