import numpy as np
import pandas as pd
from pathlib import Path
from global_land_mask import globe
import requests
from geopy.distance import geodesic
from haversine import haversine


def dropColumns(deliveryDataFrame):
    return deliveryDataFrame.drop(columns = ['ID', 'Festival', 'City', 'multiple_deliveries', 'Type_of_order', 'Vehicle_condition',])


#strips extra text
def stripEXtraText(deliveryDataFrame): 
    for col in ['Weatherconditions', 'Time_taken(min)']:
        deliveryDataFrame[col] = deliveryDataFrame[col].str.strip().str.split().str[-1]


#cast types
def castTypes(deliveryDataFrame):

    NUMERIC_FIELDS = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude','Time_taken(min)'] #id is dropped might add it again if needed

    for field in NUMERIC_FIELDS:
        if pd.api.types.is_string_dtype(deliveryDataFrame[field]):
            deliveryDataFrame[field] = pd.to_numeric(
                deliveryDataFrame[field],
                errors='coerce'
            )

#converting text nan to actual nan
def convertTextNaN(deliveryDataFrame):
    return deliveryDataFrame.replace(['nan', 'NaN', 'NAN', 'NA', 'na', 'conditions NaN'], np.nan)



#tests if the coords points to land or not
def isLand(row):
    return globe.is_land(row['Restaurant_latitude'], row['Restaurant_longitude']) and globe.is_land(row['Delivery_location_latitude'], row['Delivery_location_longitude'])


#calculates the distance
def get_distance(row):
    coords = [row["Restaurant_latitude"], row["Restaurant_longitude"], row["Delivery_location_latitude"], row["Delivery_location_longitude"]]

    if any(pd.isna(x) for x in coords):
        return float("nan")

    point1 = (row["Restaurant_latitude"], row["Restaurant_longitude"])
    point2 = (row["Delivery_location_latitude"], row["Delivery_location_longitude"])

    return haversine(point1, point2, unit="km")


#creates the distance column
def createDistances(deliveryDataFrame):

    cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']

    #fixing negative coords
    for col in cols:
        deliveryDataFrame[col]= abs(deliveryDataFrame[col])

    #dropping ocean corrds
    nRowsBefore = deliveryDataFrame.shape[0]
    print(f"number of rows before testing on land vs ocean coords: {nRowsBefore}")
    deliveryDataFrame = deliveryDataFrame[deliveryDataFrame.apply(isLand, axis=1)]
    nRowsAfter = deliveryDataFrame.shape[0]
    print(f"number of rows after testing on land vs ocean coords: {nRowsAfter}")
    print(f"{nRowsBefore-nRowsAfter} were dropped due to being on ocean\n\n")

    #dropping invalide coords
    nRowsBefore = nRowsAfter
    for col in cols:
        deliveryDataFrame[col] = deliveryDataFrame[col].where(deliveryDataFrame[col]!=0, np.nan)
    nRowsAfter = deliveryDataFrame.shape[0]
    print(f"number of rows after testing on coord equalling zero: {nRowsAfter}")
    print(f"{nRowsBefore-nRowsAfter} were dropped due to being on ocean\n\n")

    #calculate the distance 
    deliveryDataFrame["distance"] = deliveryDataFrame.apply(get_distance, axis=1)

    high_suspects = deliveryDataFrame[deliveryDataFrame['distance'] > 25]
    print(high_suspects[['distance', 'Time_taken(min)']].sort_values('distance', ascending=False))

    low_suspects = deliveryDataFrame[deliveryDataFrame['distance'] <1]
    print(low_suspects[['distance', 'Time_taken(min)']].sort_values('distance', ascending=False))

    #drop the coord columns and return the result
    return deliveryDataFrame.drop(columns=cols)


#categorizes time
def handleTime(deliveryDataFrame):
    deliveryDataFrame["Time_Orderd"] = deliveryDataFrame["Time_Orderd"].apply(bucket_time)
    return deliveryDataFrame

def bucket_time(t):
    t = pd.to_datetime(t).time()
    hour = t.hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


#calculates delivery person experience
def calculateExperience(deliveryDataFrame):

    deliveryDataFrame["Order_Date"] = pd.to_datetime(deliveryDataFrame["Order_Date"])
    first_order = deliveryDataFrame.groupby("Delivery_person_ID")["Order_Date"].transform("min")
    deliveryDataFrame["Delivery_Person_Experience_Years"] = (deliveryDataFrame["Order_Date"] - first_order).dt.days / 365.25
    deliveryDataFrame.drop(columns=["Order_Date", "Delivery_person_ID"], inplace=True)
    return deliveryDataFrame



def calculatePrepTime(order_t, pickup_t):
    order_t = pd.to_datetime(order_t, format='%H:%M:%S', errors='coerce')
    pickup_t = pd.to_datetime(pickup_t, format='%H:%M:%S', errors='coerce')

    diff = (pickup_t - order_t).dt.total_seconds() / 60
    diff = diff.where(diff >= 0, diff + 24*60)

    return diff




def handleMissingData(deliveryDataFrame):

    for col in ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance']:
        deliveryDataFrame[col] = deliveryDataFrame[col].fillna(deliveryDataFrame[col].median())

    for col in ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle']:
        deliveryDataFrame[col] = deliveryDataFrame[col].fillna(deliveryDataFrame[col].mode()[0])

    deliveryDataFrame = deliveryDataFrame.dropna(subset=['Time_Orderd'])

    #filling missing vehicle values 
    deliveryDataFrame['Type_of_vehicle'] = deliveryDataFrame['Type_of_vehicle'].fillna(deliveryDataFrame['Type_of_vehicle'].mode()[0])

    return deliveryDataFrame

    



