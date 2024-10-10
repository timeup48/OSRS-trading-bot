import requests
import csv
import time

# OSRS Grand Exchange API endpoint
API_URL = "https://prices.runescape.wiki/api/v1/osrs/latest"

# Function to fetch data from the API
def fetch_ge_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        return data['data']  # Accessing the 'data' field
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Function to export data to a CSV file
def export_to_csv(data, filename="osrs_ge_data.csv"):
    # Define the CSV headers
    headers = ['Item ID', 'High Price', 'Low Price', 'Margin', 'High Time', 'Low Time']

    # Open the CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(headers)

        # Write each item's data to the CSV
        for item_id, item_data in data.items():
            high_price = item_data.get('high', 'N/A')
            low_price = item_data.get('low', 'N/A')
            margin = high_price - low_price if isinstance(high_price, (int, float)) and isinstance(low_price, (int, float)) else 'N/A'
            high_time = item_data.get('highTime', 'N/A')
            low_time = item_data.get('lowTime', 'N/A')

            # Write row with item data
            writer.writerow([item_id, high_price, low_price, margin, high_time, low_time])

    print(f"Data successfully exported to {filename}")

# Main function to execute the process in a loop
def main():
    while True:
        # Fetch data from the API
        ge_data = fetch_ge_data()

        # Check if data is fetched successfully
        if ge_data:
            # Export data to CSV
            export_to_csv(ge_data)
        
        # Wait for 12 seconds before the next run (5 times per minute)
        time.sleep(5)

if __name__ == "__main__":
    main()
