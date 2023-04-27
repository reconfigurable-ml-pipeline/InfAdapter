import os
from joblib import dump
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from auto_tuner import AUTO_TUNER_DIRECTORY

os.system(f"mkdir -p {AUTO_TUNER_DIRECTORY}/adapter/capacity_models")

df = pd.read_csv(f"{AUTO_TUNER_DIRECTORY}/adapter/capacity_result_Feb_21.csv")
# slas = list(df["SLA"].unique())
slas = [750]

versions = [18, 34, 50, 101, 152]
models = {v: {} for v in versions}
for version in versions:
    for sla in slas:
        df_sla = df[df["SLA"] == sla]
        df_sla = df_sla.query("CPU in (1,2,4,8,16)")
        X_train = np.array(df_sla[(df_sla["ARCH"] == version)]["CPU"]).reshape(-1, 1)
        Y_train = df_sla[(df_sla["ARCH"] == version)]["capacity"]

        model = LinearRegression()
        model.fit(X_train, Y_train)
        dump(model, f"{AUTO_TUNER_DIRECTORY}/adapter/capacity_models/resnet-{version}-{sla}.joblib")
        models[version][sla] = model

for version in versions:
    for sla in slas:
        model = models[version][sla]
        df_sla = df[df["SLA"] == sla]
        X_test = np.array(df_sla[df_sla["ARCH"] == version]["CPU"]).reshape(-1, 1)

        y_pred = model.predict(X_test)
        r2s = r2_score(df_sla[df_sla["ARCH"] == version]["capacity"], y_pred)
        mse = mean_squared_error(df_sla[df_sla["ARCH"] == version]["capacity"], y_pred)
        print(f"Resnet{version}, MSE: {mse}, R2Score: {r2s}")
        
        plt.title(f"capacity prediction for resnet{version}. SLA={sla}")
        plt.plot(list(range(1, 21)), y_pred, label="prediction")
        plt.plot(list(range(1, 21)), df_sla[df_sla["ARCH"] == version]["capacity"], "go", label="profiled")
        plt.xlabel("CPU cores")
        plt.ylabel("capacity")
        plt.xticks(list(range(1, 21)))
        plt.legend()
        plt.show()
        plt.close()
