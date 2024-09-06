import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta

# Replace 'YOUR_API_KEY' with your actual TomTom API key
api_key = 'iNqASrHaEoH5Vxn8Htq1FWDrwpicPOQO'

# Define the location coordinates
latitude = 34.0317 
longitude = -118.284017

# Define the time range
end_time = int(time.time())
start_time = end_time - 900 * 1000  # Define a larger range to collect 1000 data points

# Define the interval
interval = 150 * 60  # 15 minutes in seconds

# List to store the data
traffic_data_list = []

# Collect 1000 data points
current_time = start_time
while len(traffic_data_list) < 10:
    # Construct the API URL
    url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?"
           f"point={latitude}%2C{longitude}&unit=KMPH&openLr=false&key={api_key}")

    # Make the API request
    response = requests.get(url)
    print(f"Requesting data for timestamp: {datetime.fromtimestamp(current_time)}")
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract relevant information (customize as needed)
        traffic_data = {
            'timestamp': current_time,
            'latitude': latitude,
            'longitude': longitude,
            'current_speed': data.get('flowSegmentData', {}).get('currentSpeed', None),
            'free_flow_speed': data.get('flowSegmentData', {}).get('freeFlowSpeed', None),
            'currentTravelTime': data.get('flowSegmentData', {}).get('currentTravelTime', None),
            'freeFlowTravelTime': data.get('flowSegmentData', {}).get('freeFlowTravelTime', None)
        }
        traffic_data_list.append(traffic_data)
    else:
        print("Error:", response.status_code)
    # Move to the next interval
    current_time += interval

    # To avoid hitting the API rate limits, you can add a delay between requests
    time.sleep(1)  # Sleep for 1 second (adjust as needed)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(traffic_data_list)

# Save the DataFrame to a CSV file
df.to_csv('traffic_data.csv', index=False)

print("Data collection complete. Data saved to 'traffic_data.csv'.")
