FROM python:3.8-slim

WORKDIR /app
COPY forecasters/lstm_saved_model /app/lstm_saved_model
ENV LSTM_MODEL=/app/lstm_saved_model
COPY reconfig.py reconfig.py
COPY main.py main.py
COPY notifier.py notifier.py
COPY runner.sh runner.sh
COPY capacity_models /app/capacity_models
ENV CAPACITY_MODELS=/app/capacity_models

RUN pip install --upgrade pip
RUN pip install joblib==1.2.0
RUN pip install git+ssh://git@github.com/reconfigurable-ml-pipeline/kubernetes-python-client.git
RUN pip install tensorflow==2.8.0
RUN pip install aiohttp==3.7.4
RUN pip install aioprocessing==2.0.1

ENTRYPOINT ["bash", "runner.sh"]
