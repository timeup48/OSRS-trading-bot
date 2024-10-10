simport pandas as pd
import time
from datetime import datetime
import os

# File path
input_file = 'osrs_ge_data.csv'  # Input file that is updated

# Base directory for storing item history
base_directory = 'osrs_ge_history'

# Create the base directory if it doesn't exist
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Function to append data to the output file for each item
def append_to_item_history(data):
    for index, row in data.iterrows():
        item_id = row['Item ID']
        # Create a folder for each Item ID
        item_directory = os.path.join(base_directory, str(item_id))
        os.makedirs(item_directory, exist_ok=True)  # Create the folder if it doesn't exist

        # Create a file name based on the Item ID
        output_file = os.path.join(item_directory, f'osrs_ge_history_{item_id}.csv')

        # Create output file with headers if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write('Item ID,High Price,Low Price,Margin,High Time,Low Time,Timestamp\n')

        # Append the data to the item-specific history file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{row['Item ID']},{row['High Price']},{row['Low Price']},{row['Margin']},{row['High Time']},{row['Low Time']},{timestamp}\n"
        with open(output_file, 'a') as f:
            f.write(line)

try:
    while True:
        # Read the current state of the input file
        data = pd.read_csv(input_file)

        # Append the data to the history file for each item
        append_to_item_history(data)

        # Wait for the next update (12 seconds)
        time.sleep(12)

except KeyboardInterrupt:
    print("Data collection stopped.")
