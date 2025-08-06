from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp, hour, month, dayofweek, lit, bround
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, BooleanType, FloatType
from kafka import KafkaProducer, KafkaConsumer
import json
import time
import random
from datetime import datetime, timedelta

class AccidentSeverityPredictor:
    def __init__(self, severity_model_path="models/us_accidents_severity_rf", 
                 duration_model_path="models/us_accidents_duration_rf", 
                 kafka_topic="output_topic", top_n=128):
        self.spark = SparkSession.builder.appName("AccidentSeverityPredictor").getOrCreate()
        self.severity_model = PipelineModel.load(severity_model_path)
        self.duration_model = PipelineModel.load(duration_model_path)
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
        """Veri hazırlama - hem severity hem duration tahmini için"""
        # Timestamp kolonlarından ek özellikler oluştur
        df = df.withColumn("Start_TS", to_timestamp(col("Start_Time"), "yyyy-MM-dd HH:mm:ss"))
        df = df.withColumn("Hour", hour(col("Start_TS")))
        df = df.withColumn("DayOfWeek", dayofweek(col("Start_TS")))
        df = df.withColumn("Month", month(col("Start_TS")))
        df = df.withColumn("Weather_Hour", hour(col("Start_TS")))  # Weather_Hour için Start_Time kullan
        df = df.withColumn("Weather_Day", month(col("Start_TS")))  # Weather_Day için month kullan (model eğitimindeki gibi)

        # Time slot kategorisi oluştur (0: gece, 1: sabah, 2: öğleden sonra, 3: akşam)
        df = df.withColumn("TimeSlot", 
            when((hour(col("Start_TS")) >= 0) & (hour(col("Start_TS")) < 6), 0)
            .when((hour(col("Start_TS")) >= 6) & (hour(col("Start_TS")) < 12), 1)
            .when((hour(col("Start_TS")) >= 12) & (hour(col("Start_TS")) < 18), 2)
            .otherwise(3))

        # Boolean kolonları ikili değerlere dönüştür (sadece modelde kullanılanlar)
        df = df.withColumn("Junction_Bin", when(col("Junction") == True, 1).otherwise(0))
        # Not: Crossing_Bin ve Traffic_Signal_Bin model eğitiminde kullanılmamış
        
        # Weather_Condition kolonunu temizle
        df = self._clean_column(df, "Weather_Condition")
        # City ve Street kolonlarını temizle
        df = self._clean_column(df, "City")
        df = self._clean_column(df, "Street")

        # Start_Lat ve Start_Lng yuvarlama
        df = df.withColumn("Start_Lat", bround(col("Start_Lat"), 1))
        df = df.withColumn("Start_Lng", bround(col("Start_Lng"), 1))

        # Modeller için gerekli kolonlar (model eğitimindeki feature_cols ile uyumlu)
        selected_cols = [
            "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
            "Wind_Speed(mph)", "Weather_Condition_Cleaned", "Wind_Direction",
            "Junction_Bin",  # Sadece Junction_Bin kullanılıyor
            "Civil_Twilight", "Sunrise_Sunset", "State", "City_Cleaned", "Street_Cleaned",
            "DayOfWeek", "Start_Lat", "Start_Lng", "Hour", "Month", "TimeSlot", 
            "Weather_Day", "Weather_Hour"
        ]

        return df.select(*selected_cols)

    def run_service(self):
        print("Starting prediction service...")
        while True:
            try:
                self.run_example(single_accident=True)
                time.sleep(2)  # 2 saniye bekle
            except Exception as e:
                print(f"Error in run_service: {str(e)}")
                time.sleep(1)  # Hata durumunda 1 saniye bekle

    def _get_time_based_conditions(self, hour):
        """Saate göre hava ve trafik koşullarını belirle"""
        if 5 <= hour < 7:  # Sabah erken - Daha normal koşullar
            return {
                "weather_weights": {
                    "Clear": 0.4,
                    "Fog": 0.3,
                    "Heavy Rain": 0.1,
                    "Snow": 0.1,
                    "Rain": 0.1
                },
                "traffic_factor": 1.0,  # Normal seviye
                "twilight": "Dawn",
                "risk_factor": 1.2  # Azaltıldı
            }
        elif 7 <= hour < 10:  # Sabah rush - Yoğun trafik
            return {
                "weather_weights": {
                    "Clear": 0.5,  # Daha fazla normal hava
                    "Rain": 0.2,
                    "Heavy Rain": 0.1,
                    "Fog": 0.2
                },
                "traffic_factor": 1.3,  # Azaltıldı
                "twilight": "Day",
                "risk_factor": 1.2  # Azaltıldı
            }
        elif 10 <= hour < 16:  # Gün ortası - Daha normal koşullar
            return {
                "weather_weights": {
                    "Clear": 0.6,  # Çoğunlukla güzel hava
                    "Heavy Rain": 0.1,
                    "Snow": 0.1,
                    "Fog": 0.1,
                    "Rain": 0.1
                },
                "traffic_factor": 1.0,  # Normal trafik
                "twilight": "Day",
                "risk_factor": 1.0  # Normal risk
            }
        elif 16 <= hour < 19:  # Akşam rush - Biraz daha riskli
            return {
                "weather_weights": {
                    "Clear": 0.4,
                    "Heavy Rain": 0.2,
                    "Snow": 0.1,
                    "Fog": 0.2,
                    "Rain": 0.1
                },
                "traffic_factor": 1.5,  # Azaltıldı
                "twilight": "Day",
                "risk_factor": 1.3  # Azaltıldı
            }
        elif 19 <= hour < 22:  # Akşam - Orta seviye
            return {
                "weather_weights": {
                    "Clear": 0.4,
                    "Fog": 0.2,
                    "Heavy Rain": 0.2,
                    "Snow": 0.1,
                    "Rain": 0.1
                },
                "traffic_factor": 1.1,
                "twilight": "Dusk",
                "risk_factor": 1.1
            }
        else:  # Gece - Daha normal gece koşulları
            return {
                "weather_weights": {
                    "Clear": 0.5,
                    "Fog": 0.2,
                    "Snow": 0.15,
                    "Heavy Rain": 0.15
                },
                "traffic_factor": 0.8,  # Gece az trafik
                "twilight": "Night",
                "risk_factor": 1.1
            }

    def _generate_random_accident(self):
        """Gerçekçi rastgele kaza verisi üret"""
        # Genişletilmiş şehir listesi ve koordinatları
        cities = {
            "New York": {"lat": (40.7128, 40.7818), "lng": (-74.0060, -73.9260), "state": "NY", "temp_range": (20, 95)},
            "Los Angeles": {"lat": (34.0522, 34.1522), "lng": (-118.2437, -118.1637), "state": "CA", "temp_range": (50, 95)},
            "Chicago": {"lat": (41.8781, 41.9281), "lng": (-87.6298, -87.5498), "state": "IL", "temp_range": (10, 90)},
            "Houston": {"lat": (29.7604, 29.8104), "lng": (-95.3698, -95.2898), "state": "TX", "temp_range": (40, 100)},
            "Phoenix": {"lat": (33.4484, 33.4984), "lng": (-112.0740, -111.9940), "state": "AZ", "temp_range": (45, 110)},
            "Philadelphia": {"lat": (39.9526, 40.0026), "lng": (-75.1652, -75.0852), "state": "PA", "temp_range": (25, 90)},
            "San Antonio": {"lat": (29.4241, 29.4741), "lng": (-98.4936, -98.4136), "state": "TX", "temp_range": (40, 100)},
            "San Diego": {"lat": (32.7157, 32.7657), "lng": (-117.1611, -117.0811), "state": "CA", "temp_range": (55, 85)},
            "Dallas": {"lat": (32.7767, 32.8267), "lng": (-96.7970, -96.7170), "state": "TX", "temp_range": (35, 105)},
            "Miami": {"lat": (25.7617, 25.8117), "lng": (-80.1918, -80.1118), "state": "FL", "temp_range": (60, 95)},
            "Seattle": {"lat": (47.6062, 47.6562), "lng": (-122.3321, -122.2521), "state": "WA", "temp_range": (35, 85)},
            "Denver": {"lat": (39.7392, 39.7892), "lng": (-104.9903, -104.9103), "state": "CO", "temp_range": (20, 95)},
            "Boston": {"lat": (42.3601, 42.4101), "lng": (-71.0589, -70.9789), "state": "MA", "temp_range": (15, 90)},
            "Atlanta": {"lat": (33.7490, 33.7990), "lng": (-84.3880, -84.3080), "state": "GA", "temp_range": (35, 95)},
            "San Francisco": {"lat": (37.7749, 37.8249), "lng": (-122.4194, -122.3394), "state": "CA", "temp_range": (45, 80)}
        }

        # Rastgele bir şehir seç
        city_name = random.choice(list(cities.keys()))
        city_data = cities[city_name]

        # Detaylı hava durumu seçenekleri (mevsime ve saate göre ağırlıklandırılmış)
        base_weather_conditions = {
            "Clear": ["Clear", "Sunny", "Fair", "Clear Skies"],
            "Mostly Cloudy": ["Mostly Cloudy", "Partly Cloudy", "Scattered Clouds", "Broken Clouds"],
            "Overcast": ["Overcast", "Cloudy", "Dark Skies"],
            "Light Rain": ["Light Rain", "Drizzle", "Scattered Showers"],
            "Rain": ["Rain", "Showers", "Steady Rain"],
            "Heavy Rain": ["Heavy Rain", "Thunderstorm", "Severe Thunderstorm"],
            "Light Snow": ["Light Snow", "Snow Flurries", "Light Sleet"],
            "Snow": ["Snow", "Moderate Snow", "Wintry Mix"],
            "Fog": ["Fog", "Mist", "Haze", "Low Visibility"],
            "Light Drizzle": ["Light Drizzle", "Scattered Drizzle", "Intermittent Drizzle"]
        }

        # Genişletilmiş cadde isimleri ve özellikleri
        streets = {
            "Main St": {"type": "main", "traffic_mult": 1.2},
            "Broadway": {"type": "main", "traffic_mult": 1.3},
            "1st Ave": {"type": "avenue", "traffic_mult": 1.1},
            "Park Ave": {"type": "avenue", "traffic_mult": 1.0},
            "Washington St": {"type": "main", "traffic_mult": 1.2},
            "Lake St": {"type": "residential", "traffic_mult": 0.8},
            "Market St": {"type": "commercial", "traffic_mult": 1.4},
            "Central Ave": {"type": "avenue", "traffic_mult": 1.1},
            "Madison Ave": {"type": "avenue", "traffic_mult": 1.0},
            "Highland Ave": {"type": "residential", "traffic_mult": 0.7},
            "Oak St": {"type": "residential", "traffic_mult": 0.6},
            "Maple Dr": {"type": "residential", "traffic_mult": 0.5},
            "Commerce Way": {"type": "commercial", "traffic_mult": 1.3},
            "Industrial Pkwy": {"type": "industrial", "traffic_mult": 1.1},
            "River Rd": {"type": "scenic", "traffic_mult": 0.8}
        }

        # Zaman hesaplama ve koşullar
        now = datetime.now()
        random_minutes = random.randint(0, 60)  # Son 1 saat içinde
        start_time = now - timedelta(minutes=random_minutes)
        hour = start_time.hour
        
        # Saate göre koşulları al
        conditions = self._get_time_based_conditions(hour)
        
        # Sokak seç ve trafik faktörünü uygula
        street_name = random.choice(list(streets.keys()))
        street_data = streets[street_name]
        traffic_factor = conditions["traffic_factor"] * street_data["traffic_mult"]
        
        # Trafik faktörüne göre kaza süresini ayarla
        base_duration = random.randint(15, 120)  # 15 dk ile 2 saat arası
        duration = int(base_duration * traffic_factor)  # Trafik yoğunsa süre uzar
        end_time = start_time + timedelta(minutes=duration)

        # Hava durumu seç
        weather_category = random.choices(
            list(conditions["weather_weights"].keys()),
            weights=list(conditions["weather_weights"].values())
        )[0]
        weather_condition = random.choice(base_weather_conditions[weather_category])

        # Sıcaklık aralığını şehre ve saate göre ayarla
        temp_min, temp_max = city_data["temp_range"]
        if hour < 6 or hour > 20:  # Gece
            temp_min -= 10
            temp_max -= 15
        temp = round(random.uniform(temp_min, temp_max), 1)

        # Risk faktörünü al
        risk_factor = conditions.get("risk_factor", 1.0)

        # Ekstrem koşulları azalt - daha normale yakın veriler üret
        extreme_conditions = random.random() < 0.15  # %15 olasılıkla ekstrem koşullar (önceden %40)

        # Hava durumuna ve risk faktörüne göre görüş mesafesini ayarla - daha normal değerler
        if extreme_conditions:
            visibility = round(random.uniform(0.5, 2.0), 1)  # Daha normal ekstrem değerler
        elif weather_condition in ["Fog", "Heavy Rain", "Snow"]:
            visibility = round(random.uniform(1.0, 3.0), 1)  # Daha iyi görüş
        elif weather_condition in ["Light Rain", "Light Snow", "Mist"]:
            visibility = round(random.uniform(3.0, 6.0), 1)
        else:
            visibility = round(random.uniform(5.0, 10.0), 1)  # Normal hava - iyi görüş

        # Rüzgar yönü ve hızı - daha normal değerler
        wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        wind_direction = random.choice(wind_directions)
        if extreme_conditions:
            wind_speed = round(random.uniform(25.0, 45.0), 1)  # Güçlü ama makul rüzgar
        elif "Rain" in weather_condition or "Storm" in weather_condition:
            wind_speed = round(random.uniform(15.0, 30.0), 1)  # Orta seviye rüzgar
        else:
            wind_speed = round(random.uniform(5.0, 20.0), 1)  # Normal rüzgar

        # Yağış miktarını artır
        precipitation = round(
            random.uniform(0.0, 4.0) * (2 if extreme_conditions else 1) * 
            (1.5 if "Rain" in weather_condition or "Storm" in weather_condition else 0.2),
            2
        )

        # Risk faktörüne göre ekstra özellikler - daha az ekstrem
        extra_risk = random.random() < (risk_factor * 0.1)  # Risk faktörüne bağlı olasılık (azaltıldı)
        
        # Sıcaklık ekstremlerini artır
        if extra_risk:
            if random.random() < 0.5:  # Aşırı sıcak
                temp = round(random.uniform(95, 115), 1)
            else:  # Aşırı soğuk
                temp = round(random.uniform(-10, 20), 1)

        return {
            "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "End_Time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature(F)": temp,
            "Humidity(%)": round(random.uniform(70.0, 100.0) if "Rain" in weather_condition or "Fog" in weather_condition else random.uniform(30.0, 90.0), 1),
            "Pressure(in)": round(random.uniform(28.5, 31.5), 2),  # Daha geniş basınç aralığı
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather_condition,
            "Wind_Direction": wind_direction,
            "Civil_Twilight": conditions["twilight"],
            "Sunrise_Sunset": "Day" if 6 <= hour < 20 else "Night",
            "State": city_data["state"],
            "City": city_name,
            "Street": street_name,
            "Wind_Chill(F)": round(temp - (wind_speed * 0.7), 1),
            "Junction": random.choice([True, False]),
            "Crossing": random.choices([True, False], weights=[0.7 if street_data["type"] in ["main", "commercial"] else 0.3, 0.3 if street_data["type"] in ["main", "commercial"] else 0.7])[0],
            "Traffic_Signal": random.choices([True, False], weights=[0.8 if street_data["type"] in ["main", "commercial", "avenue"] else 0.2, 0.2 if street_data["type"] in ["main", "commercial", "avenue"] else 0.8])[0],
            "Start_Lat": round(random.uniform(city_data["lat"][0], city_data["lat"][1]), 6),
            "Start_Lng": round(random.uniform(city_data["lng"][0], city_data["lng"][1]), 6)
        }



    def run_example(self, single_accident=False):
        # Tek kaza verisi üret
        example_rows = [self._generate_random_accident()]
        if not single_accident:
            additional_accidents = random.randint(2, 6)
            example_rows.extend([self._generate_random_accident() for _ in range(additional_accidents)])
        
        df = self.spark.createDataFrame(example_rows)
        
        # Veriyi hazırla (hem severity hem duration için aynı veriyi kullan)
        df_clean = self._prepare_data(df)
        
        # Debug: Veri kontrol
        print("=== DEBUG INFO ===")
        print("DataFrame shape:", df_clean.count(), "x", len(df_clean.columns))
        print("Columns:", df_clean.columns)
        df_clean.show(5, truncate=False)
        
        # Severity ve Duration tahminlerini yap
        severity_predictions = self.severity_model.transform(df_clean)
        duration_predictions = self.duration_model.transform(df_clean)

        # Debug: Tahmin sonuçları
        print("\n=== PREDICTIONS ===")
        print("Severity predictions:")
        severity_predictions.select("prediction", "probability").show(truncate=False)
        print("Duration predictions:")
        duration_predictions.select("prediction", "probability").show(truncate=False)

        # Tahminleri al
        severity_data = severity_predictions.select("prediction").collect()
        duration_data = duration_predictions.select("prediction").collect()
        
        # Duration tahminini sınıflara göre dakikaya çevir
        def duration_class_to_minutes(prediction):
            if prediction == 0:  # Very Short
                return 5
            elif prediction == 1:  # Short
                return 15
            elif prediction == 2:  # Medium
                return 60
            elif prediction == 3:  # Long
                return 180
            else:  # Very Long
                return 360
                
        # Duration sınıf isimlerini al
        def duration_class_to_name(prediction):
            if prediction == 0:
                return "Very Short"
            elif prediction == 1:
                return "Short"
            elif prediction == 2:
                return "Medium"
            elif prediction == 3:
                return "Long"
            else:
                return "Very Long"
        
        # Orijinal verileri al
        original_data = df.select(
            "Start_Lat", "Start_Lng", "Temperature(F)", 
            "Weather_Condition", "Start_Time", "City", "Street"
        ).collect()
        
        # Tüm verileri birleştir
        for sev, dur, orig in zip(severity_data, duration_data, original_data):
            # Duration modelinin prediction'ını kullan (0-4 arası sınıf)
            duration_class = int(dur["prediction"])
            duration_minutes = duration_class_to_minutes(duration_class)
            duration_name = duration_class_to_name(duration_class)
            
            message = {
                "severity": float(sev["prediction"]),
                "duration": duration_minutes,  # Dakika cinsinden süre
                "duration_name": duration_name,  # Sınıf ismi (Short, Medium, etc.)
                "location": {
                    "lat": float(orig["Start_Lat"]),
                    "lng": float(orig["Start_Lng"])
                },
                "timestamp": time.time(),
                "details": {
                    "temperature": float(orig["Temperature(F)"]),
                    "weather": str(orig["Weather_Condition"]),
                    "time": str(orig["Start_Time"]),
                    "estimated_duration": duration_name,  # Sınıf ismini göster
                    "duration_minutes": f"{int(duration_minutes)} minutes",  # Dakika bilgisi ayrı
                    "duration_class": duration_class,  # Debug için ekle
                    "city": str(orig["City"]) if "City" in orig else "Unknown",
                    "street": str(orig["Street"]) if "Street" in orig else "Unknown"
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
