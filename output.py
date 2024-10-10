import os
import pandas as pd
import time
import joblib
import numpy as np
from datetime import datetime

# Configuration
DATA_FILE = 'osrs_ge_data.csv'
MODEL_DIR = '/Volumes/PARROT/untitled folder/item_models'
BASE_DATA_PATH = '/Volumes/PARROT/untitled folder/osrs_ge_history/'
SELL_LOG_PATH = '/Volumes/PARROT/untitled folder/sell_log.csv'
STARTING_CASH = 1000000  # Starting cash for the simulation
MAX_ITEMS_PER_PURCHASE = 100  # Limit the maximum number of items that can be purchased at once

if os.path.exists(SELL_LOG_PATH):
    pd.DataFrame(columns=['Item_ID', 'Sell_Price', 'Sell_Volume', 'Sell_Timestamp']).to_csv(SELL_LOG_PATH, index=False)

def is_file_valid(filepath):
    """Check if the file exists and is non-empty."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def load_models():
    """Load all trained models from the model directory."""
    models = {}
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith('.pkl'):
            item_id = model_file.split('_')[-1].split('.')[0]
            models[item_id] = joblib.load(os.path.join(MODEL_DIR, model_file))
    return models

def load_model(item_id):
    """Load a specific model for an item ID."""
    model_file = os.path.join(MODEL_DIR, f'model_item_{item_id}.pkl')
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print(f"Model for Item ID {item_id} not found.")
        return None

def get_all_item_ids(base_data_path):
    """List all directories in the specified path and use them as item IDs."""
    return [folder for folder in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, folder))]

def fetch_latest_data(item_id):
    """Fetch the latest data for an item."""
    item_file = os.path.join(BASE_DATA_PATH, str(item_id), f'osrs_ge_history_{item_id}.csv')
    if os.path.exists(item_file):
        return pd.read_csv(item_file)
    else:
        print(f"Data file for Item ID {item_id} not found.")
        return None

def calculate_profitability(item_id, data, purchase_price):
    """Calculate the profitability of an item based on recent data."""
    recent_data = data.tail(1)
    current_high = recent_data['High Price'].values[0]
    return (current_high - purchase_price) if purchase_price < current_high else 0

def get_top_profitable_items(all_item_ids, max_cash):
    """Get the top profitable items that can be purchased with available cash."""
    top_profitable_items = []
    cannot_afford_count = 0  # Counter for items that cannot be afforded

    for item_id in all_item_ids:
        data = fetch_latest_data(item_id)
        if data is None or data.empty:
            continue
        
        purchase_price = data['Low Price'].iloc[-1]
        
        if purchase_price <= 0:
            continue  # Skip if price is zero or negative
        
        max_quantity = max_cash // purchase_price
        if max_quantity > 0:  # Only consider if at least one item can be purchased
            # Cap the maximum quantity that can be purchased
            purchase_quantity = min(max_quantity, MAX_ITEMS_PER_PURCHASE)
            profit = (data['High Price'].iloc[-1] - purchase_price) * purchase_quantity
            top_profitable_items.append((item_id, purchase_price, profit))
        else:
            cannot_afford_count += 1  # Increment if can't afford

    top_profitable_items.sort(key=lambda x: x[2], reverse=True)
    print(f"Items that cannot be afforded: {cannot_afford_count}")
    return top_profitable_items[:20]

def purchase_items(max_cash, min_margin, max_price):
    """Purchase items based on available cash, minimum margin, and maximum price."""
    purchased_items = []
    all_item_ids = get_all_item_ids(BASE_DATA_PATH)  # Fetch item IDs

    for item_id in all_item_ids:
        data = fetch_latest_data(item_id)
        if data is None or data.empty:
            continue
            
        price = data['Low Price'].iloc[-1]

        if price > max_price:
            continue  # Skip if exceeds max price
            
        quantity = max_cash // price
        if quantity > 0:
            # Cap the maximum quantity that can be purchased
            purchase_quantity = min(quantity, MAX_ITEMS_PER_PURCHASE)
            margin = calculate_profitability(item_id, data, price)
            if margin >= min_margin:
                total_cost = purchase_quantity * price
                if total_cost <= max_cash:
                    purchased_items.append((item_id, purchase_quantity, price))
                    print(f"Purchased {purchase_quantity} of Item ID {item_id} at {price} GP each.")
                    max_cash -= total_cost
    return purchased_items

def decide_sell(item_id, model, data, quantity, min_margin):
    """Determine whether to sell an item based on predictions and conditions."""
    recent_data = data.tail(1)
    current_high = recent_data['High Price'].values[0]
    current_low = recent_data['Low Price'].values[0]
    current_margin = current_high - current_low

    # Example feature calculations:
    previous_high = data['High Price'].iloc[-2]  # Previous day's high price
    previous_low = data['Low Price'].iloc[-2]    # Previous day's low price
    
    # Create features DataFrame with actual values
    features = pd.DataFrame({
        'High Price': [current_high],
        'Low Price': [current_low],
        'Margin': [current_margin],
        # Only include features that were used during training
    })

    # Ensure the features match what was used during training
    if 'Average Price Last 5 Days' in features.columns:
        features = features.drop(columns=['Average Price Last 5 Days'])
    if 'Price Change Percentage Last Day' in features.columns:
        features = features.drop(columns=['Price Change Percentage Last Day'])
    if 'Volume Traded Last Day' in features.columns:
        features = features.drop(columns=['Volume Traded Last Day'])

    predicted_change = model.predict(features)[0]

    # Check conditions for selling
    if (predicted_change > 0 and quantity > 0 and (current_margin >= min_margin)):
        log_sell(item_id, current_high + predicted_change, quantity)  # Sell at predicted price
        return True, current_high + predicted_change
    
    return False, None

def log_sell(item_id, price, volume):
    """Log the selling transaction in a CSV file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry = pd.DataFrame({
        'Item_ID': [item_id],
        'Sell_Price': [price],
        'Sell_Volume': [volume],
        'Sell_Timestamp': [timestamp]
    })
    new_entry.to_csv(SELL_LOG_PATH, mode='a', header=False, index=False)
    print(f"Sold {volume} of Item ID {item_id} at {price:.2f} GP. Logged at {timestamp}.")

def monitor_sell_opportunities(purchased_items, min_margin):
    """Monitor selling opportunities for purchased items."""
    while True:
        for item_id, quantity, purchase_price in purchased_items:
            model = load_model(item_id)
            if model is None:
                print(f"Model for Item ID {item_id} not found. Skipping.")
                continue

            data = fetch_latest_data(item_id)
            if data is None or data.shape[0] < 2:
                print(f"Insufficient data for Item ID {item_id}. Skipping.")
                continue

            should_sell, sell_price = decide_sell(item_id, model, data)
            if should_sell:
                print(f"Ready to sell Item ID {item_id} at {sell_price:.2f} GP each.")

        time.sleep(60)  # Check every minute

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
    cannot_afford_count = 0  # Counter for items that cannot be afforded

    for _, row in df.iterrows():
        item_id = str(row['Item ID'])
        high_price = row['High Price']
        low_price = row['Low Price']
        margin = row['Margin']
        
        if item_id not in models:
            print(f"No model found for Item ID {item_id}.")
            continue
        
        model = models[item_id]
        purchase_price = low_price
        
        # Check if can afford to purchase the item
        max_quantity = cash // purchase_price
        
        if max_quantity > 0:
            purchase_quantity = min(max_quantity, MAX_ITEMS_PER_PURCHASE)
            cash -= purchase_quantity * purchase_price
            inventory[item_id] = inventory.get(item_id, 0) + purchase_quantity
            print(f"Bought {purchase_quantity} of Item ID {item_id} at {purchase_price} GP each.")

            # Track items that cannot be afforded
        else:
            cannot_afford_count += 1

    # Check sell opportunities for purchased items
    for item_id, quantity in inventory.items():
        if quantity > 0:
            data = fetch_latest_data(item_id)
            if data is None or data.empty:
                continue
            
            should_sell, sell_price = decide_sell(item_id, models[item_id], data, quantity, min_margin)
            if should_sell:
                cash += sell_price * quantity
                profit += (sell_price - purchase_price) * quantity
                print(f"Sold {quantity} of Item ID {item_id} at {sell_price} GP each.")

    print(f"Final cash: {cash}, Total profit: {profit} GP.")
    print(f"Items that cannot be afforded: {cannot_afford_count}")

if __name__ == "__main__":
    min_margin = 1000  # Define your minimum margin
    max_price = 20000  # Define your maximum purchase price

    purchased_items = purchase_items(STARTING_CASH, min_margin, max_price)
    monitor_sell_opportunities(purchased_items, min_margin)
