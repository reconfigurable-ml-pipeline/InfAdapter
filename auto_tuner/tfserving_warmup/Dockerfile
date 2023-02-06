FROM python:3.8-slim

WORKDIR /app

RUN apt-get -y update

RUN pip install --upgrade pip
RUN pip install tensorflow-cpu==2.11.0
RUN pip install --no-deps tensorflow-serving-api==2.11.0
RUN pip install grpcio==1.43.0

COPY cat.jpg cat.jpg
COPY warmup.py warmup.py

ENTRYPOINT ["python", "warmup.py"]
