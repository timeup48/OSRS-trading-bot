import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DATA_PATH = 'osrs_ge_history'
MODEL_DIR = 'item_models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Check if the file exists and is not empty
def is_file_valid(filepath):
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return False
    elif os.path.getsize(filepath) == 0:
        print(f"File is empty: {filepath}")
        return False
    return True

# Feature engineering function
def feature_engineering(df):
    df['Price_Change'] = df['High Price'].diff()
    df['Margin'] = df['High Price'] - df['Low Price']
    df['MA3_High'] = df['High Price'].rolling(window=3).mean()
    df['MA10_High'] = df['High Price'].rolling(window=10).mean()
    df['Volatility'] = df['Price_Change'].pct_change()
    df.dropna(inplace=True)
    return df

# Train the model for a specific item
import numpy as np

# Inside train_model function, after feature engineering and before splitting the data
def train_model(item_id):
    item_file = os.path.join(BASE_DATA_PATH, str(item_id), f'osrs_ge_history_{item_id}.csv')

    if not is_file_valid(item_file):
        print(f"Data file is invalid or empty: {item_file}. Skipping training for Item ID {item_id}.")
        return

    try:
        df = pd.read_csv(item_file)
    except pd.errors.EmptyDataError:
        print(f"No data found in {item_file}. Skipping training for Item ID {item_id}.")
        return
    except Exception as e:
        print(f"Error reading {item_file}: {e}. Skipping training for Item ID {item_id}.")
        return

    if df.shape[0] < 2:
        print(f"Not enough data for Item ID {item_id}. Skipping...")
        return

    # Feature engineering
    df = feature_engineering(df)
    
    # Check for NaN, infinite, or extremely large values and remove them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if df.shape[0] < 2:
        print(f"Not enough valid data after cleaning for Item ID {item_id}. Skipping...")
        return

    # Prepare features and target
    X = df[['High Price', 'Low Price', 'Margin', 'MA3_High', 'MA10_High', 'Volatility']]
    y = df['Price_Change']

    if len(X) < 2:
        print(f"Not enough samples to train model for Item ID {item_id}. Skipping...")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0:
        print(f"Training set is empty for Item ID {item_id}. Skipping...")
        return

    # Normalize data (handling NaN or Inf is done before this step)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Item ID {item_id} - MSE: {mse}, MAE: {mae}, R²: {r2}")

    # Save model
    model_filename = os.path.join(MODEL_DIR, f'model_item_{item_id}.pkl')
    joblib.dump(model, model_filename)
    print(f"Model for Item ID {item_id} saved as {model_filename}.")

    # Visualize price changes
    visualize_price_changes(df, item_id)

# Visualize price changes for an item
def visualize_price_changes(df, item_id):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Timestamp'], df['High Price'], label='High Price', color='blue', marker='o')
    plt.plot(df['Timestamp'], df['Low Price'], label='Low Price', color='orange', marker='o')
    plt.fill_between(df['Timestamp'], df['Low Price'], df['High Price'], color='skyblue', alpha=0.3)
    plt.title(f'Price Changes for Item ID {item_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_filename = os.path.join(BASE_DATA_PATH, str(item_id), f'price_trends_{item_id}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Price trend visualization saved for Item ID {item_id} as {plot_filename}.")

# Monitor and retrain models when data changes
def monitor_data():
    item_ids = []

    for item_dir in os.listdir(BASE_DATA_PATH):
        item_path = os.path.join(BASE_DATA_PATH, item_dir)
        if os.path.isdir(item_path):
            item_ids.append(item_dir)

    last_modified = {item_id: 0 for item_id in item_ids}

    while True:
        time.sleep(10)

        for item_id in item_ids:
            item_file = os.path.join(BASE_DATA_PATH, str(item_id), f'osrs_ge_history_{item_id}.csv')
            current_modified = os.path.getmtime(item_file)

            if current_modified != last_modified[item_id]:
                print(f"Data file for Item ID {item_id} updated. Retraining model...")
                train_model(item_id)
                last_modified[item_id] = current_modified

if __name__ == "__main__":
    monitor_data()
