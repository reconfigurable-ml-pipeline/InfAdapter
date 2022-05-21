FROM python:3.8-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==1.10.1
RUN pip install torchvision==0.11.2
RUN pip install kserve==0.8.0

COPY predictor.py predictor.py

ENTRYPOINT ["python", "predictor.py"]