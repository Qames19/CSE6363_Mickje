import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from Layer_Implementations.Sequential import Sequential
from Layer_Implementations.Linear import Linear
from Layer_Implementations.Relu import Relu
from Layer_Implementations.MeanSquaredError import MSELoss
from Layer_Implementations.BinaryCrossEntropy import BinaryCrossEntropy

# Define 3 different models


def compute_trip_duration(df):
    """
    Computes trip duration in seconds from pickup and dropoff times.
    """
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["calculated_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    return df

# This is the formula/implementation that ChatGPT gave to me when asking how to calc distance as crow flies from coordinates.
# I have further modified it to switch between freedom units and kilometers
# d = 2R * sin^-1(sqrt(sin^2*(delta(phi)/2) + cos(phi_1)cos(phi_2)sin^2(delta(lambda)/2)))
def haversine_distance(lat1, lon1, lat2, lon2, unit="miles"):
    """
    Computes Haversine distance between two sets of coordinates.
    - unit="miles" → Returns miles
    - unit="km" → Returns kilometers
    """
    R = 3958.8 if unit == "miles" else 6371  # Earth radius
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def compute_distances(df):
    """
    Computes both Haversine distances (miles & km) between pickup and dropoff points.
    """
    df["distance_miles"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
        unit="miles"
    )
    
    df["distance_km"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
        unit="km"
    )
    
    return df

def extract_time_features(df):
    """
    Extracts useful time-based features from pickup_datetime.
    """
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Drop original timestamps (not needed after feature extraction)
    df.drop(columns=["pickup_datetime", "dropoff_datetime"], inplace=True)
    
    return df

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_features(X_train, X_test):
    numerical_features = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "distance_miles",
        "day_of_week", "hour_sin", "hour_cos"
    ]

    # Ensure only valid columns are selected
    available_features = [col for col in numerical_features if col in X_train.columns]

    # Check for infinite or missing values
    if not np.isfinite(X_train[available_features]).all().all() or not np.isfinite(X_test[available_features]).all().all():
        raise ValueError("X_train or X_test contains infinite or missing values.")

    # Fit scaler on combined data (to ensure consistent scaling)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pd.concat([X_train[available_features], X_test[available_features]]))

    # Apply transformation correctly
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[available_features] = scaler.transform(X_train[available_features])
    X_test_scaled[available_features] = scaler.transform(X_test[available_features])

    return X_train_scaled, X_test_scaled

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, loss_function):
    training_losses = []
    validation_losses = []
    best_val_loss = float("inf")
    patience = 3  # Stop training after 3 epochs of no improvement
    patience_counter = 0

    # Convert X_train and X_val to numpy arrays if they are DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()

    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Ensure X_batch and y_batch are numpy arrays
            if isinstance(X_batch, pd.DataFrame):
                X_batch = X_batch.to_numpy()
            if isinstance(y_batch, pd.DataFrame):
                y_batch = y_batch.to_numpy()

            # Debug: Verify batch dimensions and types
            print(f"X_batch shape: {X_batch.shape}, type: {type(X_batch)}")
            print(f"y_batch shape: {y_batch.shape}, type: {type(y_batch)}")

            # Forward pass
            predictions = model.forward(X_batch)

            # Ensure predictions are numpy arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.asarray(predictions)

            # Compute loss
            loss = loss_function.forward(y_batch, predictions)

            # Check for invalid values
            if np.isnan(loss) or np.isinf(loss):
                print(f"Detected NaN or Inf loss at epoch {epoch}. Stopping training.")
                return
            
            # Update loss
            total_loss += loss

            # Backward pass
            grad_loss = loss_function.backward()
            model.backward(grad_loss)

            # Set lambda_reg
            lambda_reg = 0.01
            # Update weights
            for layer in model.layers:
                if hasattr(layer, "weights") and hasattr(layer, "bias"):
                    np.clip(layer.grad_weights, -1, 1, out=layer.grad_weights)
                    np.clip(layer.grad_bias, -1, 1, out=layer.grad_bias)

                    layer.weights -= lr * (layer.grad_weights + lambda_reg * layer.weights)
                    layer.bias -= lr * layer.grad_bias

        # Compute validation loss
        val_predictions = model.forward(X_val)

        # Ensure val_predictions are numpy arrays
        if not isinstance(val_predictions, np.ndarray):
            val_predictions = np.asarray(val_predictions)

        val_loss = loss_function.forward(y_val, val_predictions)

        training_losses.append(total_loss / len(X_train))
        validation_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {training_losses[-1]:.6f}, Validation Loss: {val_loss:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered! Stopping training.")
            break

    # Plot loss curves
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    return training_losses, validation_losses

# def main():
    # # initialize input_file
    # input_file = os.path.join("Datasets", "nyc_taxi_data.npy")
# 
    # # Check for dataset
    # if not os.path.exists(input_file):
        # print(f"{input_file} does not exist!")
        # sys.exit(1)
    # 
    # # Load dataset
    # dataset = np.load(input_file, allow_pickle=True).item()
    # X_train, y_train = dataset["X_train"], dataset["y_train"]
    # X_test, y_test = dataset["X_test"], dataset["y_test"]
# 
    # # Perform pre-processing
    # # Compute trip duration (only on training set)
    # X_train = compute_trip_duration(X_train)
# 
    # # Compute distances (in both miles and km)
    # X_train = compute_distances(X_train)
    # X_test  = compute_distances(X_test)
# 
    # # Extract time features
    # X_train = extract_time_features(X_train)
    # X_test  = extract_time_features(X_test)
# 
    # # One-hot encode vendor_id
    # X_train = pd.get_dummies(X_train, columns=["vendor_id"])
    # X_test  = pd.get_dummies(X_test,  columns=["vendor_id"])
# 
    # # Filter out 0 passengers, trips less than 30 seconds, and trips less than 0.1 mile for X_train only
    # X_train = X_train[(X_train["passenger_count"] > 0) & (X_train["calculated_duration"] >= 30) & (X_train["distance_miles"] >= 0.1)]
    # y_train = y_train.loc[X_train.index]
# 
    # # Filter out 0 passengers for X_test
    # X_test = X_test[(X_test["passenger_count"] > 0)]
    # y_test = y_test.loc[X_test.index]
    # 
    # # Drop the column "id"
    # X_train.drop(columns=["id"], inplace=True)
    # X_test.drop(columns=["id"], inplace=True) 
# 
    # X_train.drop(columns=["passenger_count", "store_and_fwd_flag", "calculated_duration", "hour", "month", "distance_km", "vendor_id_2"], inplace=True)
    # X_test.drop(columns=["passenger_count", "store_and_fwd_flag", "hour", "month", "distance_km", "vendor_id_2"], inplace=True)
# 
    # # import seaborn as sns
    # # correlation_matrix = X_train.corr()
    # # plt.figure(figsize=(10,8))
    # # sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    # # plt.title("Feature Correlation Matrix")
    # # plt.show()    
    # # sys.exit(0)
# 
    # # Scale X Train and Test features
    # X_train, X_test = scale_features(X_train, X_test)
    # 
    # # Apply log transformation on y data. ChatGPT recommendation
    # y_train = np.log1p(y_train).to_numpy().reshape(-1, 1)
    # y_test = np.log1p(y_test).to_numpy().reshape(-1, 1)
# 
    # # Split data for training
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# 
    # # Define input_size
    # input_size = X_train.shape[1]
# 
    # # Model 1: Small network
    # model_1 = Sequential()
    # model_1.add(Linear(input_size, 32))
    # model_1.add(Relu())
    # model_1.add(Linear(32, 1))
    # 
    # # Model 2: Medium network (current model)
    # model_2 = Sequential()
    # model_2.add(Linear(input_size, 32))
    # model_2.add(Relu())
    # model_2.add(Linear(32, 16))
    # model_2.add(Relu())
    # model_2.add(Linear(16, 1))
    # 
    # # Model 3: Larger network
    # model_3 = Sequential()
    # model_3.add(Linear(input_size, 64))
    # model_3.add(Relu())
    # model_3.add(Linear(64, 32))
    # model_3.add(Relu())
    # model_3.add(Linear(32, 16))
    # model_3.add(Relu())
    # model_3.add(Linear(16, 1))
# 
    # models = [model_1, model_2, model_3]
    # model_names = ["Small Model", "Medium Model", "Large Model"]
# 
    # for model, name in zip(models, model_names):
        # print(f"\n Training {name}")
        # train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001, loss_function=MSELoss())
# 
    # # Evaluate test set performance
    # for model, name in zip(models, model_names):
        # test_predictions = model.forward(X_test)
        # loss_function = MSELoss()
        # test_loss = loss_function.forward(y_test, test_predictions)
    # 
        # print(f"\n{name} - Test Loss (RMSLE): {test_loss:.6f}")
    # 
        # # Compare with benchmark (0.513 RMSLE)
        # if test_loss < 0.513:
            # print("Your model outperforms the benchmark!")
        # else:
            # print("Your model did not beat the benchmark.")

def main():
    # Initialize input_file
    input_file = os.path.join("Datasets", "nyc_taxi_data.npy")

    # Check for dataset
    if not os.path.exists(input_file):
        print(f"{input_file} does not exist!")
        sys.exit(1)
    
    # Load dataset
    dataset = np.load(input_file, allow_pickle=True).item()
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]

    # Perform pre-processing
    X_train = compute_trip_duration(X_train)
    X_train = compute_distances(X_train)
    X_test  = compute_distances(X_test)
    X_train = extract_time_features(X_train)
    X_test  = extract_time_features(X_test)
    X_train = pd.get_dummies(X_train, columns=["vendor_id"])
    X_test  = pd.get_dummies(X_test,  columns=["vendor_id"])

    # Filter out 0 passengers, trips less than 30 seconds, and trips less than 0.1 mile for X_train only
    X_train = X_train[(X_train["passenger_count"] > 0) & (X_train["calculated_duration"] >= 30) & (X_train["distance_miles"] >= 0.1)]
    y_train = y_train.loc[X_train.index.tolist()]

    # Filter out 0 passengers for X_test
    X_test = X_test[(X_test["passenger_count"] > 0)]
    y_test = y_test.loc[X_test.index.tolist()]
    
    # Drop the column "id"
    X_train.drop(columns=["id"], inplace=True)
    X_test.drop(columns=["id"], inplace=True)
    X_train.drop(columns=["passenger_count", "store_and_fwd_flag", "calculated_duration", "hour", "month", "distance_km", "vendor_id_2"], inplace=True)
    X_test.drop(columns=["passenger_count", "store_and_fwd_flag", "hour", "month", "distance_km", "vendor_id_2"], inplace=True)

    # Debug: Verify shapes and types after preprocessing
    print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
    print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
    print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
    print(f"y_test shape: {y_test.shape}, type: {type(y_test)}")

    # Scale X Train and Test features
    X_train, X_test = scale_features(X_train, X_test)

    # Apply log transformation on y data
    # y_train = np.log1p(y_train).to_numpy().reshape(-1, 1)
    # y_test = np.log1p(y_test).to_numpy().reshape(-1, 1)

    # Split data for training
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define input_size
    input_size = X_train.shape[1]

    # Define models
    model_1 = Sequential()
    model_1.add(Linear(input_size, 32))
    model_1.add(Relu())
    model_1.add(Linear(32, 1))

    model_2 = Sequential()
    model_2.add(Linear(input_size, 32))
    model_2.add(Relu())
    model_2.add(Linear(32, 16))
    model_2.add(Relu())
    model_2.add(Linear(16, 1))

    model_3 = Sequential()
    model_3.add(Linear(input_size, 64))
    model_3.add(Relu())
    model_3.add(Linear(64, 32))
    model_3.add(Relu())
    model_3.add(Linear(32, 16))
    model_3.add(Relu())
    model_3.add(Linear(16, 1))

    models = [model_1, model_2, model_3]
    model_names = ["Small Model", "Medium Model", "Large Model"]

    for model, name in zip(models, model_names):
        print(f"\n Training {name}")
        train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001, loss_function=BinaryCrossEntropy())

    # Evaluate test set performance
    for model, name in zip(models, model_names):
        test_predictions = model.forward(X_test)

        # Debug: Verify test_predictions
        print(f"test_predictions type: {type(test_predictions)}, shape: {test_predictions.shape}")

        loss_function = BinaryCrossEntropy()
        test_loss = loss_function.forward(y_test, test_predictions)

        print(f"\n{name} - Test Loss (RMSLE): {test_loss:.6f}")

        if test_loss < 0.513:
            print("Your model outperforms the benchmark!")
        else:
            print("Your model did not beat the benchmark.")

if __name__ == "__main__":
    main()