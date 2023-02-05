FROM python:3.8-slim

WORKDIR /app

RUN apt-get -y update

RUN pip install --upgrade pip
RUN pip install tensorflow==2.8.0
RUN pip install tensorflow-serving-api==2.8.0
RUN pip install grpcio==1.39.0

COPY cat.jpg cat.jpg
COPY warmup.py warmup.py

ENTRYPOINT ["python", "warmup.py"]
