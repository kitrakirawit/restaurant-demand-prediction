import pandas as pd

visit_df = pd.read_csv("air_visit_data.csv")
store_df = pd.read_csv("air_store_info.csv")
date_df = pd.read_csv("date_info.csv")

merged_df = (
    visit_df.merge(store_df, on="air_store_id", how="left")
    .merge(date_df, left_on="visit_date", right_on="calendar_date", how="left")
)

merged_df = merged_df[
    [
        "air_store_id",
        "visit_date",
        "visitors",
        "day_of_week",
        "holiday_flg",
        "air_genre_name",
        "air_area_name",
    ]
]

merged_df.to_csv("merged_restaurant_data.csv", index=False)
print(" Merged file created successfully!")
