FROM bitnami/spark:3.3.0

USER root
RUN pip install kafka-python numpy pandas scikit-learn

COPY . /app
WORKDIR /app

CMD ["spark-submit", "AccidentSeverityPredictor.py"]
