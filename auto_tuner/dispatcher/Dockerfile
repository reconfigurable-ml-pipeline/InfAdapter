FROM python:3.8-slim

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install aiohttp==3.7.4
COPY main.py main.py

ENTRYPOINT ["python", "main.py"]
