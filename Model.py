import os
import pandas as pd
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define paths to datasets and images
# Change these paths to match your local environment
labels_path_2022 = '2022/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv'
labels_path_2023 = '2023/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv'
base_image_path_2023_validation = '2023_validacion/2023/Satellite'
base_image_path_2023_test = 'Test/Satellite'

# Function to load and preprocess TIF images
def load_and_preprocess_tif(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.moveaxis(image, 0, -1)  # Move band axis to last dimension
        image = np.array(image, dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())  # Normalize
        image = tf.image.resize(image, [128, 128])  # Resize to 128x128
    return image

# Function to load images and labels from the dataset
def load_images_and_labels(data):
    images = []
    labels = []
    for _, row in data.iterrows():
        image = load_and_preprocess_tif(row['image_path'])
        if image is not None:
            images.append(image)
            labels.append(row['yield_per_acre'])
    return np.array(images), np.array(labels)

# Load yield data for 2022 and 2023
labels_df_2022 = pd.read_csv(labels_path_2022)
labels_df_2023 = pd.read_csv(labels_path_2023)

# Build image paths based on CSV data
def build_image_path(row, base_path):
    location = row['location']
    block = f"TP{int(row['block'])}"  # Assuming 'block' is a number converted to TP1, TP2, etc.
    experiment = str(int(row['experiment']))
    row_value = str(int(row['row']))
    range_value = str(int(row['range']))

    image_name = f"{location}-{block}-{experiment}_{range_value}_{row_value}.TIF"
    image_path = os.path.join(base_path, block, image_name)

    print(f"Constructing image path: {image_path}")
    return image_path

# Function to split data into training, validation, and test sets
def split_data(data, test_size=0.2, val_size=0.1):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)
    return train_data, val_data, test_data

# Create CNN model for regression
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='linear')  # Regression output to predict yield
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Load and prepare data for satellite images
satellite_data_2022 = labels_df_2022.copy()
satellite_data_2023 = labels_df_2023.copy()

# Combine datasets from both years
combined_data = pd.concat([satellite_data_2022, satellite_data_2023], ignore_index=True)

# Filter data to remove records with NaN yield values
filtered_data = combined_data.dropna(subset=['yieldPerAcre'])

# Split satellite data into training, validation, and test sets
satellite_train, satellite_val, satellite_test = split_data(filtered_data)
satellite_train_imgs, satellite_train_labels = load_images_and_labels(satellite_train)
satellite_val_imgs, satellite_val_labels = load_images_and_labels(satellite_val)
satellite_test_imgs, satellite_test_labels = load_images_and_labels(satellite_test)

# Train the CNN model using the training dataset
input_shape = (128, 128, 6)  # 6 channels in the TIF images
satellite_cnn_model = create_cnn_model(input_shape)
satellite_cnn_model.fit(satellite_train_imgs, satellite_train_labels, validation_data=(satellite_val_imgs, satellite_val_labels), epochs=10, batch_size=32)

# Evaluate the model on the test dataset
test_loss, test_rmse = satellite_cnn_model.evaluate(satellite_test_imgs, satellite_test_labels)
print(f"RMSE on satellite test data: {test_rmse}")

# Update CSV with predictions for validation data
validation_csv_path = '2023_validacion/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv'
validation_csv_data = pd.read_csv(validation_csv_path)
base_image_path = base_image_path_2023_validation

predictions = []
for _, row in validation_csv_data.iterrows():
    image_path = build_image_path(row, base_image_path)
    image = load_and_preprocess_tif(image_path)
    if image is not None:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predicted_yield = satellite_cnn_model.predict(image)[0][0]  # Make prediction
        predictions.append(predicted_yield)
    else:
        predictions.append(np.nan)  # Handle case where image was not found

validation_csv_data['yieldPerAcre'] = predictions
validation_csv_data.to_csv(validation_csv_path, index=False)

# Update CSV with predictions for test data
test_csv_path = 'Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv'
test_csv_data = pd.read_csv(test_csv_path)
base_image_path = base_image_path_2023_test

predictions = []
for _, row in test_csv_data.iterrows():
    image_path = build_image_path(row, base_image_path)
    image = load_and_preprocess_tif(image_path)
    if image is not None:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predicted_yield = satellite_cnn_model.predict(image)[0][0]  # Make prediction
        predictions.append(predicted_yield)
    else:
        predictions.append(np.nan)  # Handle case where image was not found

test_csv_data['yieldPerAcre'] = predictions
test_csv_data.to_csv(test_csv_path, index=False)
