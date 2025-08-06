from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, BooleanType, FloatType
from kafka import KafkaProducer, KafkaConsumer
import json
import time

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
        df = df \
            .withColumn("Start_TS", to_timestamp(col("Start_Time"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("End_TS", to_timestamp(col("End_Time"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("Duration", ((unix_timestamp(col("End_TS")) - unix_timestamp(col("Start_TS"))) / 60).cast(DoubleType())) \
            .drop("Start_TS", "End_TS", "Start_Time", "End_Time")

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

    def run_example(self):
        print("Starting prediction process...")
        # Minimal example input
        example_rows = [
            {
                "Start_Time": "2022-09-01 08:00:00",
                "End_Time": "2022-09-01 09:00:00",
                "Temperature(F)": 70.0,
                "Humidity(%)": 55.0,
                "Pressure(in)": 29.5,
                "Visibility(mi)": 10.0,
                "Wind_Speed(mph)": 5.0,
                "Precipitation(in)": 0.0,
                "Weather_Condition": "Clear",
                "Wind_Direction": "NW",
                "Civil_Twilight": "Day",
                "Sunrise_Sunset": "Day",
                "State": "CA",
                "City": "Los Angeles",
                "Street": "Sunset Blvd",
                "Wind_Chill(F)": 70.0,
                "Junction": False,
                "Crossing": True,
                "Traffic_Signal": True
            },
            {
                "Start_Time": "2022-09-01 10:00:00",
                "End_Time": "2022-09-01 11:30:00",
                "Temperature(F)": 60.0,
                "Humidity(%)": 80.0,
                "Pressure(in)": 30.0,
                "Visibility(mi)": 8.0,
                "Wind_Speed(mph)": 10.0,
                "Precipitation(in)": 0.1,
                "Weather_Condition": "Rain",
                "Wind_Direction": "SE",
                "Civil_Twilight": "Day",
                "Sunrise_Sunset": "Day",
                "State": "NY",
                "City": "New York",
                "Street": "Broadway",
                "Wind_Chill(F)": 59.0,
                "Junction": True,
                "Crossing": False,
                "Traffic_Signal": True
            }
        ]

        print("Creating DataFrame...")
        df = self.spark.createDataFrame(example_rows)
        print("Preparing data...")
        df_clean = self._prepare_data(df)
        print("Making predictions...")
        predictions = self.model.transform(df_clean)
        print("Predictions made successfully.")

        # Add sample coordinates for visualization
        locations = [
            {"lat": 40.7128, "lng": -74.0060},  # New York
            {"lat": 34.0522, "lng": -118.2437}  # Los Angeles
        ]

        output_data = predictions.select("prediction").collect()
        
        for i, row in enumerate(output_data):
            message = {
                "prediction": float(row["prediction"]),
                "location": locations[i % len(locations)],
                "timestamp": time.time()
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
