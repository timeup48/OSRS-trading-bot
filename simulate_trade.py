import os
import pandas as pd
import time
import joblib

# Path to the live CSV file and the model directory
DATA_FILE = 'osrs_ge_data.csv'
MODEL_DIR = 'item_models'
STARTING_CASH = 1000000  # Starting cash for the simulation

def is_file_valid(filepath):
    """Check if the file exists and is non-empty."""
    if not os.path.exists(filepath):
        return False
    elif os.path.getsize(filepath) == 0:
        return False
    return True

def load_models():
    """Load all trained models from the model directory."""
    models = {}
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith('.pkl'):
            item_id = model_file.split('_')[-1].split('.')[0]
            models[item_id] = joblib.load(os.path.join(MODEL_DIR, model_file))
    return models

def simulate_trade():
    """Simulate buying and selling of items based on predicted prices."""
    if not is_file_valid(DATA_FILE):
        print(f"Data file {DATA_FILE} is invalid or empty.")
        return
    
    models = load_models()
    
    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        print(f"No data found in {DATA_FILE}. Exiting simulation.")
        return
    except Exception as e:
        print(f"Error reading {DATA_FILE}: {e}. Exiting simulation.")
        return

    cash = STARTING_CASH
    inventory = {}
    profit = 0

    for _, row in df.iterrows():
        item_id = str(row['Item ID'])
        high_price = row['High Price']
        low_price = row['Low Price']
        margin = row['Margin']
        
        if item_id not in models:
            continue
        
        model = models[item_id]
        predicted_price_change = model.predict([[high_price, low_price, margin]])[0]

        # Simulate buy if price is predicted to increase
        if predicted_price_change > 0 and cash >= low_price:
            num_items_to_buy = cash // low_price
            inventory[item_id] = inventory.get(item_id, 0) + num_items_to_buy
            cash -= num_items_to_buy * low_price
            print(f"Bought {num_items_to_buy} of Item ID {item_id} at {low_price} gp each.")

        # Simulate sell if price is predicted to decrease and we have inventory
        elif predicted_price_change < 0 and inventory.get(item_id, 0) > 0:
            sale_price = high_price
            cash += inventory[item_id] * sale_price
            profit += (sale_price - low_price) * inventory[item_id]
            print(f"Sold {inventory[item_id]} of Item ID {item_id} at {sale_price} gp each for a profit of {profit} gp.")
            inventory[item_id] = 0  # Reset inventory after selling

    print(f"Simulation complete. Final cash: {cash}, Total profit: {profit}.")

def monitor_and_simulate():
    """ Continuously monitor the CSV file for changes and simulate trades. """
    last_modified_time = 0
    while True:
        if os.path.exists(DATA_FILE):
            current_modified_time = os.path.getmtime(DATA_FILE)
            if current_modified_time != last_modified_time:
                print(f"Detected changes in {DATA_FILE}. Running simulation...")
                simulate_trade()
                last_modified_time = current_modified_time

        user_input = input("Type 'stop' to end the simulation, or press Enter to continue: ")
        if user_input.lower() == 'stop':
            print("Stopping simulation.")
            break
        time.sleep(10)  # Check for file updates every 10 seconds

if __name__ == "__main__":
    monitor_and_simulate()
