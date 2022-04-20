FROM python:3.8-slim
WORKDIR /app

# Install python packages
RUN pip install --upgrade pip
RUN pip install torch==1.10.1
RUN pip install torchvision==0.11.2
RUN pip install seldon-core==1.12.0
RUN pip install Flask==1.1.1
RUN pip install Jinja2==3.0.3
RUN pip install itsdangerous==2.0.1
RUN pip install Werkzeug==2.0.2

# Copy source code
COPY ClassificationModel.py .

# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME ClassificationModel
ENV SERVICE_TYPE MODEL

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE