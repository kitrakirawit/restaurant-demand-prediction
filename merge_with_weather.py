import pandas as pd

#Loading the two data sets
restaurants=pd.read_csv("merged_restaurant_data.csv")
weather=pd.read_csv("tokyo_weather.csv")

# Rename data in the weather file to match with restaurant data
weather=weather.rename(columns={"date":"visit_date"})

#Merge the two tables
merged= pd.merge(restaurants,weather, on="visit_date", how="left")

#Save
merged.to_csv("restaurant_with_weather.csv",index=False)

print("Merged")
print("Shape:", merged.shape)
print(merged.head())