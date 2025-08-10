# Uber-Ride-Analysis

This project focuses on analyzing Uber ride data to uncover trends in ride requests, cancellations, trip durations, and customer behaviors.
It involves a complete data analytics pipeline — from raw data to actionable insights — using Python, Pandas, NumPy, Matplotlib, and Seaborn.

# Step 1: Load the Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
url = "https://gitlab.crio.do/me_notebook/me_jupyter_uberanalysis/-/raw/master/Uber_ride_analysis_dataset.csv"
df = pd.read_csv(url)

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# Step 2: Standardize Column Names

print("\nOriginal Column Names:", df.columns.tolist())
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("Standardized Column Names:", df.columns.tolist())

# Step 3: Handle Missing Values

print("\nMissing values before handling:\n", df.isnull().sum())

# Drop rows where trip_status or trip_cost are missing
df = df.dropna(subset=['trip_status', 'trip_cost'])

# Fill missing driver_id with -1
df['driver_id'] = df['driver_id'].fillna(-1)

# Fill missing payment_method with mode
payment_mode = df['payment_method'].mode()[0]
df['payment_method'] = df['payment_method'].fillna(payment_mode)

# Keep missing timestamps only for cancelled/failed trips
completed_trips = df['trip_status'].str.lower() == 'completed'
df.loc[completed_trips, 'start_timestamp'] = df.loc[completed_trips, 'start_timestamp'].fillna(method='ffill')
df.loc[completed_trips, 'drop_timestamp'] = df.loc[completed_trips, 'drop_timestamp'].fillna(method='ffill')

print("\nMissing values after handling:\n", df.isnull().sum())

# Step 4: Convert Columns to Proper Types

date_cols = ['request_timestamp', 'start_timestamp', 'drop_timestamp']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print("\nData types after conversion:\n", df.dtypes)

# Step 5: Identify Numeric Columns

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric Columns:", numeric_cols)

# Step 6: Outlier Detection and Treatment

for col in ['trip_cost', 'extra_tip']:
    # Plot before
    plt.boxplot(df[col].dropna())
    plt.title(f"{col} distribution")
    plt.show()

  # Calculate IQR bounds
  Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

  # Cap outliers
  df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

  # Plot after
   plt.boxplot(df[col].dropna())
    plt.title(f"{col} distribution - Outliers Handled")
    plt.show()

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/87fbacf4-5569-422b-841f-4b54ff993361" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2924498b-39a1-41b5-a7dc-e1e0722aa2c2" />

  # Step 7: Summary Statistics

print("\nSummary Statistics:\n", df.describe())

# Step 8

df.shape

# Step 9

df.dtypes

# Step 10: Visualize Trip Status Distribution

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
trip_status_counts = df['trip_status'].value_counts()
sns.barplot(x=trip_status_counts.index, y=trip_status_counts.values, palette="viridis")
plt.title("Trip Status Distribution")
plt.xlabel("Trip Status")
plt.ylabel("Number of Trips")
plt.xticks(rotation=45)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a08f87ad-ecd7-416c-a404-5aae964f8844" />

# Step 11: Visualize Trip Cost Distribution

plt.figure(figsize=(6, 4))
plt.hist(df['trip_cost'], bins=range(0, int(df['trip_cost'].max()) + 100, 100), edgecolor='black')
plt.title("Trip Cost Distribution")
plt.xlabel("Trip Cost")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c7eac41f-a8b3-4175-8eb7-2ab6bb43398c" />

# Step 12: trip_cost_stats 

trip_cost_stats = (
    df.groupby('payment_method')['trip_cost']
      .agg(['mean', 'median', 'count'])
      .sort_values(by='mean', ascending=False)
)
print("Trip Cost Analysis by Payment Method:\n", trip_cost_stats)

# Step 13: Calculate Trip Duration
# Ensure datetime columns are in proper format

df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], errors='coerce')
df['drop_timestamp'] = pd.to_datetime(df['drop_timestamp'], errors='coerce')

# Calculate duration in minutes
df['trip_duration_min'] = (df['drop_timestamp'] - df['start_timestamp']).dt.total_seconds() / 60
print("Trip Duration Column Added.")

# Step 14: Visualize Trip Duration Distribution

plt.figure(figsize=(6, 4))
plt.hist(df['trip_duration_min'].dropna(), bins=range(0, int(df['trip_duration_min'].max()) + 10, 10), edgecolor='black')
plt.title("Trip Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ecb74d0c-9596-4eee-a77c-f1566b5e4a46" />

# Step 15: Calculate Total Cost of the Trip
# Assuming trip_cost and extra_tip columns exist

df['total_cost'] = df['trip_cost'] + df['extra_tip']
print("'total_cost' column added.")
df[['trip_cost', 'extra_tip', 'total_cost']].head()


# Step 16: Convert Time Columns to DateTime Format
time_columns = ['request_timestamp', 'start_timestamp', 'drop_timestamp']

for col in time_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print("Time columns converted to datetime.")
df[time_columns].dtypes

# Step 17: Extract Date and Time Components
# Request Date/Day/Time/Hour

df['request_date'] = df['request_timestamp'].dt.date
df['request_day'] = df['request_timestamp'].dt.day_name()
df['request_time'] = df['request_timestamp'].dt.time
df['request_hour'] = df['request_timestamp'].dt.hour

# Start Date/Day/Time/Hour
df['start_date'] = df['start_timestamp'].dt.date
df['start_day'] = df['start_timestamp'].dt.day_name()
df['start_time'] = df['start_timestamp'].dt.time
df['start_hour'] = df['start_timestamp'].dt.hour

# Drop Date/Day/Time/Hour
df['drop_date'] = df['drop_timestamp'].dt.date
df['drop_day'] = df['drop_timestamp'].dt.day_name()
df['drop_time'] = df['drop_timestamp'].dt.time
df['drop_hour'] = df['drop_timestamp'].dt.hour

print("Date, day, time, and hour components extracted.")
df[['request_date', 'request_day', 'request_time', 'request_hour']].head()


# Step 18: Calculating the Ride Delay

df['ride_delay'] = (df['start_timestamp'] - df['request_timestamp']).dt.total_seconds() / 3600
print("'ride_delay' column (in hours) calculated.")
df[['request_timestamp', 'start_timestamp', 'ride_delay']].head()

 # Step 19: Calculating the cancellation_reason
 
 df['cancellation_reason'] = np.where(
    (df['driver_id'] == -1) & (df['trip_status'] == 'No Cars Available'), 'No Cabs',
    np.where(
        (df['driver_id'] == -1) & (df['trip_status'] == 'Trip Cancelled'), 'Passenger',
        np.where(
            (df['driver_id'] != -1) & (df['trip_status'] == 'Trip Cancelled'), 'Driver',
            'Trip Completed'
        )
    )
)

print("'cancellation_reason' column added.")
df[['driver_id', 'trip_status', 'cancellation_reason']].head()

# Step 20: Create New DataFrame for Analysis

cols_for_analysis = [
    'request_id', 'driver_id', 'trip_status', 'request_day', 'request_hour',
    'start_day', 'start_hour', 'drop_day', 'drop_hour', 'ride_delay',
    'trip_cost', 'weather', 'cancellation_reason'
]

new_df = df[cols_for_analysis].copy()
print("New DataFrame created with selected columns.")
new_df.head()

# Step 21: Bar Chart → Request Count Vs Day

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
sns.countplot(data=new_df, x='request_day', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
], palette='Set1')
plt.title("Request Count Vs Day", fontsize=14)
plt.xlabel("Day of the Week")
plt.ylabel("Number of Requests")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bded2efb-057b-4ee6-b0e0-336b7068c35d" />

# Step 22: Bar Chart → Request Count Vs Hour

plt.figure(figsize=(12, 5))
sns.countplot(data=new_df, x='request_hour', palette='coolwarm')
plt.title("Request Count Vs Hour", fontsize=14)
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Requests")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6f44159d-2d3c-4d5c-b7d9-1ac4cacf673d" />

# Step 23: Trip Status Bifurcation (Percentage)

trip_status_rates = new_df['trip_status'].value_counts(normalize=True) * 100
print("Trip Status Bifurcation (in %):\n", trip_status_rates)

# Step 24: Pie Chart → Trip Status Bifurcation

plt.figure(figsize=(8, 8))
plt.pie(trip_status_rates.values, labels=trip_status_rates.index,
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title("Trip Status Bifurcation", fontsize=14)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/aec9ff57-4eb0-464a-86ae-e53b2c45b9ae" />

# Step 25: Identify Who Cancels the Most Rides
# Filter for cancelled trips and count reasons

cancellation_trends = new_df[new_df['trip_status'] == 'Trip Cancelled']['cancellation_reason'].value_counts()
print("Cancellation Trends:\n", cancellation_trends)

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(cancellation_trends.values, labels=cancellation_trends.index,
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title("Trip Cancellation Trend", fontsize=14)
plt.show()

# (Optional) Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=cancellation_trends.index, y=cancellation_trends.values, palette='Set2')
plt.title("Trip Cancellation Trend", fontsize=14)
plt.xlabel("Cancellation Reason")
plt.ylabel("Number of Cancellations")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3f5d3a06-6456-4b4a-a423-3d4477f5f7be" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a9edc8f1-1b52-4fa3-a699-03a68b8bafd3" />

# Step 26: Create New DataFrame for Incomplete Rides

incomplete_df = df[df['trip_status'] != 'Trip Completed'][[
    'request_id', 'pickup_point', 'drop_point', 'driver_id',
    'trip_status', 'payment_method', 'weather',
    'request_day', 'request_hour', 'cancellation_reason'
]].copy()

print(f"Incomplete rides dataframe created — {incomplete_df.shape[0]} rows")
incomplete_df.head()

#Step 27: Proportion of Incomplete Ride Statuses (Pie Chart)

import seaborn as sns
import matplotlib.pyplot as plt

status_counts = incomplete_df['trip_status'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(status_counts.values, labels=status_counts.index,
        autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('pastel'))
plt.title("Proportion of Incomplete Ride Statuses", fontsize=14)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4c16579d-29c1-4633-993e-b057f1e47469" />

# Step 28: Stacked Histogram → Incomplete Rides by Day & Reason

plt.figure(figsize=(12, 6))
sns.histplot(data=incomplete_df, x='request_day', hue='cancellation_reason',
             multiple='stack', palette='Set2', shrink=0.8)
plt.title("Incomplete Rides by Day and Reason", fontsize=14)
plt.xlabel("Day")
plt.ylabel("Number of Incomplete Rides")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/58ad0821-be05-4065-9287-e20023e35d4f" />


# Step 29: Stacked Histogram → Incomplete Rides by Hour & Reason

plt.figure(figsize=(12, 6))
sns.histplot(data=incomplete_df, x='request_hour', hue='cancellation_reason',
             multiple='stack', palette='Set1', shrink=0.8)
plt.title("Incomplete Rides by Hour and Reason", fontsize=14)
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Incomplete Rides")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dc8ae4cb-14f8-4d97-9db2-482947939999" />

# Step 30: Relationship Between Weather & Cancellation Reason

weather_cancellation = (
    incomplete_df.groupby(['weather', 'cancellation_reason'])
    .size()
    .reset_index(name='count')
)

plt.figure(figsize=(12, 6))
sns.barplot(data=weather_cancellation, x='weather', y='count',
            hue='cancellation_reason', palette='coolwarm')
plt.title('Cancellations by Weather and Reason per Ride Type', fontsize=14)
plt.xlabel('Weather Conditions')
plt.ylabel('Count of Cancellations')
plt.legend(title='Cancellation Reason')
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ed906c40-ea9f-4921-9640-14492a9553ab" />













