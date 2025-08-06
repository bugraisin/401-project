
# Accident Severity Prediction

## Prerequisites
- Docker & Docker Compose installed
- Node.js and npm installed (version 16+ recommended)


## 1. Start Kafka + Spark Predictor (Docker)

From `/backend` folder, run:

```bash
docker-compose up --build
```

This will:
- Start Zookeeper + Kafka brokers
- Build and run PySpark predictor container that loads the ML model and produces predictions to Kafka

---

## 2. Run NestJS Frontend (Consumer)

From `/frontend` folder:

```bash
npm install
npm run start
```

The backend will:
- Connect to Kafka at `localhost:9092`
- Consume predictions from topic `output_topic`
- Broadcast updates over WebSocket (see `AccidentsGateway`)



## 3. Access Frontend

By default:

```
http://localhost:3000
```



## Notes

- Make sure Docker is running and using **Linux containers**
- Kafka topic `output_topic` is auto-created by the predictor container on startup
- NestJS app expects Kafka on localhost:9092; adjust if needed in `AccidentConsumerService`



## Troubleshooting

- If NestJS fails to connect to Kafka, wait for Kafka container to be fully ready, then restart backend
- To manually create Kafka topic:

```bash
docker exec -it <kafka-container-name> kafka-topics --create --topic output_topic --partitions 1 --replication-factor 1 --if-not-exists --bootstrap-server localhost:9092
```

- For Docker issues on Windows, ensure Linux containers mode is enabled



## Stopping

```bash
docker-compose down
```

```bash
# in /frontend
npm run stop  # or ctrl+C
```