# %%
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, to_timestamp, unix_timestamp
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline


spark = SparkSession.builder \
    .appName("US Accidents Severity Prediction") \
    .master("local[*]") \
    .getOrCreate()


# %%
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("dataset.csv")

# %%
df2 = df \
    .drop("ID","Source", "Zipcode", "Timezone", "Airport_Code", "Amenity",
          "Bump", "Give_Way", "No_Exit", "Railway", "Description", "County",
          "Roundabout", "Station", "Stop", "Nautical_Twilight", "Astronomical_Twilight", "Country")

# %%
df3 = df2 \
    .withColumn("Start_TS", to_timestamp(col("Start_Time"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("End_TS", to_timestamp(col("End_Time"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("Duration", ((unix_timestamp(col("End_TS")) - unix_timestamp(col("Start_TS"))) / 60).cast(DoubleType())) \
    .drop("Start_TS", "End_TS", "Start_Time", "End_Time")

df3.show()

# %%
selected_cols = [ "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Crossing", "Traffic_Signal",
      "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition", "Wind_Direction", "Junction", "Duration","Severity",
      "Civil_Twilight", "Sunrise_Sunset", "State", "City_Cleaned", "Wind_Chill(F)",  "Street_Cleaned" ]


# Hangi sütunları işleyeceğimizi tanımla
columns_to_clean = ["City", "Street"]
top_n = 128


# Sık geçen değerleri belirleyip "Other" ile gruplayan fonksiyon
def clean_column(df, column_name, top_n=128):
    top_values_df = df.groupBy(column_name).count().orderBy(col("count").desc()).limit(top_n)
    top_values_list = [row[column_name] for row in top_values_df.collect()]
   
    cleaned_col_name = f"{column_name}_Cleaned"
    df = df.withColumn(
        cleaned_col_name,
        when(col(column_name).isin(top_values_list), col(column_name)).otherwise("Other")
    )
    return df


# %%
from pyspark.sql.functions import col

# Sınıfları ayır
df_sev2 = df3.filter(col("Severity") == 2)
df_others = df3.filter(col("Severity") != 2)

# Severity=2 olanları kırp (örneğin %30'unu al)
df_sev2_downsampled = df_sev2.sample(withReplacement=False, fraction=0.3, seed=42)

# Veri kümesini yeniden oluştur
df3 = df_sev2_downsampled.union(df_others)

df3.show()

# %%
# Her sütun için işlemi uygula
for col_name in columns_to_clean:
    df3 = clean_column(df3, col_name, top_n=top_n)
    
df_selected = df3.select(*selected_cols)

categorical_cols = [
    "Weather_Condition", "Wind_Direction", "Civil_Twilight",
    "Sunrise_Sunset", "State", "City_Cleaned", "Street_Cleaned"
]


indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_Idx", handleInvalid="keep")
    for col in categorical_cols
]

feature_cols = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)", "Wind_Chill(F)", "Traffic_Signal",
    "Weather_Condition_Idx", "Wind_Direction_Idx", "Civil_Twilight_Idx",
    "Sunrise_Sunset_Idx", "State_Idx", "City_Cleaned_Idx", "Street_Cleaned_Idx"
]


assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)


rf = RandomForestClassifier(
    labelCol="Severity",
    featuresCol="features",
    numTrees=20,
    maxBins=216
)

df_selected.show()

# %%
# 9. NA temizliği ve veri bölme
df_no_na = df_selected.dropna().cache()
train, test = df_no_na.randomSplit([0.8, 0.2], seed=42)

df_no_na.show()

# %%
# 8. Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])
model = pipeline.fit(df_no_na)

# %%
model.write().overwrite().save("models/us_accidents_severity_rf")

# %%
# 11. Tahmin üret
predictions = model.transform(test)
df_no_na.unpersist()

evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction")

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


