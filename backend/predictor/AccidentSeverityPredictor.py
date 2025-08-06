from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, BooleanType, FloatType
from kafka import KafkaProducer, KafkaConsumer
import json
import time
import random
from datetime import datetime, timedelta

class AccidentSeverityPredictor:
    def __init__(self, model_path="models/us_accidents_severity_rf", kafka_topic="output_topic", top_n=128):
        self.spark = SparkSession.builder.appName("AccidentSeverityPredictor").getOrCreate()
        self.model = PipelineModel.load(model_path)
        self.kafka_topic = kafka_topic
        self.top_n = top_n

        self.producer = KafkaProducer(
            bootstrap_servers='kafka:29092',  # Docker internal network address
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def _clean_column(self, df, column_name):
        top_values_df = df.groupBy(column_name).count().orderBy(col("count").desc()).limit(self.top_n)
        top_values_list = [row[column_name] for row in top_values_df.collect()]
        cleaned_col_name = f"{column_name}_Cleaned"
        df = df.withColumn(
            cleaned_col_name,
            when(col(column_name).isin(top_values_list), col(column_name)).otherwise("Other")
        )
        return df

    def _prepare_data(self, df):
        # Zaman sütunlarını işle
        df = df.withColumn("Duration", 
            ((unix_timestamp(col("End_Time")) - unix_timestamp(col("Start_Time"))) / 60).cast(DoubleType()))

        df = self._clean_column(df, "City")
        df = self._clean_column(df, "Street")

        selected_cols = [
            "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Duration",
            "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition", "Wind_Direction",
            "Civil_Twilight", "Sunrise_Sunset", "State", "City_Cleaned", "Street_Cleaned",
            "Wind_Chill(F)", "Junction", "Crossing", "Traffic_Signal"
        ]

        return df.select(*selected_cols)

    def run_service(self):
        print("Starting prediction service...")
        while True:
            try:
                print("Running predictions...")
                self.run_example()
                print("Sleeping for 10 seconds...")
                time.sleep(10)
            except Exception as e:
                print(f"Error in run_service: {str(e)}")
                time.sleep(5)  # Wait a bit before retrying

    def _generate_random_accident(self):
        """Gerçekçi rastgele kaza verisi üret"""
        # Büyük şehirler ve koordinatları
        cities = {
            "New York": {"lat": (40.7128, 40.7818), "lng": (-74.0060, -73.9260), "state": "NY"},
            "Los Angeles": {"lat": (34.0522, 34.1522), "lng": (-118.2437, -118.1637), "state": "CA"},
            "Chicago": {"lat": (41.8781, 41.9281), "lng": (-87.6298, -87.5498), "state": "IL"},
            "Houston": {"lat": (29.7604, 29.8104), "lng": (-95.3698, -95.2898), "state": "TX"},
            "Phoenix": {"lat": (33.4484, 33.4984), "lng": (-112.0740, -111.9940), "state": "AZ"},
            "Philadelphia": {"lat": (39.9526, 40.0026), "lng": (-75.1652, -75.0852), "state": "PA"},
            "San Antonio": {"lat": (29.4241, 29.4741), "lng": (-98.4936, -98.4136), "state": "TX"},
            "San Diego": {"lat": (32.7157, 32.7657), "lng": (-117.1611, -117.0811), "state": "CA"},
            "Dallas": {"lat": (32.7767, 32.8267), "lng": (-96.7970, -96.7170), "state": "TX"},
            "San Jose": {"lat": (37.3382, 37.3882), "lng": (-121.8863, -121.8063), "state": "CA"}
        }

        # Rastgele bir şehir seç
        city_name = random.choice(list(cities.keys()))
        city_data = cities[city_name]
        
        # Hava durumu seçenekleri ve olasılıkları
        weather_conditions = {
            "Clear": 0.3,
            "Mostly Cloudy": 0.2,
            "Overcast": 0.1,
            "Light Rain": 0.1,
            "Rain": 0.1,
            "Heavy Rain": 0.05,
            "Light Snow": 0.05,
            "Snow": 0.03,
            "Fog": 0.05,
            "Light Drizzle": 0.02
        }

        # Zaman hesaplama
        now = datetime.now()
        random_minutes = random.randint(0, 60)  # Son 1 saat içinde
        start_time = now - timedelta(minutes=random_minutes)
        duration = random.randint(15, 120)  # 15 dk ile 2 saat arası
        end_time = start_time + timedelta(minutes=duration)

        # Ana cadde isimleri
        streets = ["Main St", "Broadway", "1st Ave", "Park Ave", "Washington St", "Lake St", 
                  "Market St", "Central Ave", "Madison Ave", "Highland Ave"]

        return {
            "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "End_Time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature(F)": round(random.uniform(20.0, 100.0), 1),
            "Humidity(%)": round(random.uniform(30.0, 100.0), 1),
            "Pressure(in)": round(random.uniform(29.0, 31.0), 2),
            "Visibility(mi)": round(random.uniform(0.5, 10.0), 1),
            "Wind_Speed(mph)": round(random.uniform(0.0, 30.0), 1),
            "Precipitation(in)": round(random.uniform(0.0, 2.0), 2),
            "Weather_Condition": random.choices(list(weather_conditions.keys()), 
                                             weights=list(weather_conditions.values()))[0],
            "Wind_Direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "Civil_Twilight": random.choice(["Day", "Night"]),
            "Sunrise_Sunset": random.choice(["Day", "Night"]),
            "State": city_data["state"],
            "City": city_name,
            "Street": random.choice(streets),
            "Wind_Chill(F)": round(random.uniform(20.0, 95.0), 1),
            "Junction": random.choice([True, False]),
            "Crossing": random.choice([True, False]),
            "Traffic_Signal": random.choice([True, False]),
            "Start_Lat": round(random.uniform(city_data["lat"][0], city_data["lat"][1]), 6),
            "Start_Lng": round(random.uniform(city_data["lng"][0], city_data["lng"][1]), 6)
        }

    def run_example(self):
        print("Starting prediction process...")
        # 3-7 arası rastgele kaza üret
        num_accidents = random.randint(3, 7)
        example_rows = [self._generate_random_accident() for _ in range(num_accidents)]

        print("Creating DataFrame...")
        df = self.spark.createDataFrame(example_rows)
        print("Preparing data...")
        df_clean = self._prepare_data(df)
        print("Making predictions...")
        predictions = self.model.transform(df_clean)
        print("Predictions made successfully.")

        # Tahminleri al
        predictions_data = predictions.select("prediction").collect()
        # Orijinal verileri al
        original_data = df.select(
            "Start_Lat", "Start_Lng", "Temperature(F)", 
            "Weather_Condition", "Start_Time"
        ).collect()
        
        # İki veri setini birleştir
        for pred, orig in zip(predictions_data, original_data):
            message = {
                "prediction": float(pred["prediction"]),
                "location": {
                    "lat": float(orig["Start_Lat"]),
                    "lng": float(orig["Start_Lng"])
                },
                "timestamp": time.time(),
                "details": {
                    "temperature": float(orig["Temperature(F)"]),
                    "weather": str(orig["Weather_Condition"]),
                    "time": str(orig["Start_Time"])
                }
            }
            self.producer.send(self.kafka_topic, message)
            print("Sent to Kafka:", message)

        self.producer.flush()
        print("All messages sent.")


if __name__ == "__main__":
    predictor = AccidentSeverityPredictor()
    try:
        predictor.run_service()
    except KeyboardInterrupt:
        print("Service stopped by user")
    except Exception as e:
        print(f"Service stopped due to error: {str(e)}")
