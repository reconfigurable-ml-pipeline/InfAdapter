FROM python:3.8-slim

WORKDIR /app

RUN apt-get -y update && apt-get -y install git

RUN pip install --upgrade pip
RUN pip install scikit-learn==1.2.0
RUN pip install git+https://github.com/reconfigurable-ml-pipeline/kubernetes-python-client.git
RUN pip install tensorflow-cpu==2.11.0
RUN pip install aiohttp==3.8.3
RUN pip install aioprocessing==2.0.1
RUN pip install pandas==1.5.2

COPY forecasters/lstm_saved_model /app/lstm_saved_model
ENV LSTM_MODEL=/app/lstm_saved_model
COPY reconfig.py reconfig.py
COPY main.py main.py
COPY notifier.py notifier.py
COPY runner.sh runner.sh
COPY capacity_models /app/capacity_models
ENV CAPACITY_MODELS=/app/capacity_models

ENTRYPOINT ["bash", "runner.sh"]
