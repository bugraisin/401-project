import findspark
findspark.init()

from AccidentSeverityPredictor import AccidentSeverityPredictor

veri = [
    {
        "Start_Time": "2023-12-10 07:00:00",
        "End_Time": "2023-12-10 07:45:00",
        "Temperature(F)": 15.0,
        "Humidity(%)": 85.0,
        "Pressure(in)": 29.5,
        "Visibility(mi)": 2.0,
        "Wind_Speed(mph)": 20.0,
        "Precipitation(in)": 0.4,
        "Weather_Condition": "Snow",
        "Wind_Direction": "N",
        "Civil_Twilight": "Night",
        "Sunrise_Sunset": "Night",
        "State": "IL",
        "Junction": True,
        "Traffic_Signal": False,
        "Crossing": True,
        "City": "Chicago",
        "Street": "W Adams St",
        "Wind_Chill(F)": 10.0
    },
    {
        "Start_Time": "2024-03-20 18:30:00",
        "End_Time": "2024-03-20 18:40:00",
        "Temperature(F)": 85.0,
        "Humidity(%)": 30.0,
        "Pressure(in)": 30.2,
        "Visibility(mi)": 10.0,
        "Wind_Speed(mph)": 3.0,
        "Precipitation(in)": 0.0,
        "Weather_Condition": "Clear",
        "Wind_Direction": "SE",
        "Civil_Twilight": "Day",
        
        "Sunrise_Sunset": "Day",
        "State": "TX",
        "Junction": False,
        "Traffic_Signal": True,
        "Crossing": False,
        "City": "Houston",
        "Street": "Main St",
        "Wind_Chill(F)": 84.0
    },
    {
        "Start_Time": "2023-09-01 05:00:00",
        "End_Time": "2023-09-01 06:00:00",
        "Temperature(F)": 40.0,
        "Humidity(%)": 95.0,
        "Pressure(in)": 29.3,
        "Visibility(mi)": 1.0,
        "Wind_Speed(mph)": 15.0,
        "Precipitation(in)": 1.0,
        "Weather_Condition": "Heavy Rain",
        "Wind_Direction": "W",
        "Civil_Twilight": "Night",
        "Sunrise_Sunset": "Night",
        "State": "FL",
        "Junction": True,
        "Traffic_Signal": True,
        "Crossing": True,
        "City": "Miami",
        "Street": "Biscayne Blvd",
        "Wind_Chill(F)": 39.0
    }
]


predictor = AccidentSeverityPredictor("models/us_accidents_severity_rf")

predictor.predict_from_rows(    
    rows=veri,
    show_columns=["City_Cleaned", "Street_Cleaned", "prediction"],
    n=10
)

predictor.stop()
