#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# =============================================================================
# PART 1: Data Preparation

def preprocess_for_cnn(image_path, target_size=(224, 224)):
    """
    Read an image, resize, convert to RGB, and normalize.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def load_images(image_dir, target_size=(224, 224)):
    """
    Load all images from a directory and preprocess them.
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    filenames = []
    
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        try:
            image = preprocess_for_cnn(img_path, target_size)
            images.append(image)
            filenames.append(filename)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
    
    images = np.array(images)
    return filenames, images

def load_dataset(csv_file, image_dir, target_size=(224, 224)):
    """
    Load images and labels from a CSV file.
    """
    df = pd.read_csv(csv_file)
    images, labels = [], []
    
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        try:
            image = preprocess_for_cnn(img_path, target_size)
            images.append(image)
            labels.append([row['warp_count'], row['weft_count']])
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels, dtype='float32')
    return images, labels

# =============================================================================
# PART 2: Build and Train the Model

def build_fabric_cnn(input_shape=(224, 224, 3)):
    """
    Build a CNN model for regression.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)  # Output: warp and weft counts
    ])
    
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the CNN model.
    """
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return history

def load_trained_model(model_path):
    """
    Load the trained model.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

# =============================================================================
# PART 3: Predict and Display Results

def predict_fabric_counts(model, image_dir, output_csv='predictions.csv'):
    """
    Predict warp and weft counts for all images and save results.
    """
    filenames, images = load_images(image_dir)
    if len(images) == 0:
        print("No valid images found.")
        return
    
    predictions = model.predict(images)
    results_df = pd.DataFrame({
        'filename': filenames,
        'predicted_warp_count': predictions[:, 0],
        'predicted_weft_count': predictions[:, 1]
    })
    
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # Display images with predictions
    num_samples = min(10, len(images))
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, (num_samples + 1) // 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"Warp: {predictions[i, 0]:.1f}\nWeft: {predictions[i, 1]:.1f}")
    plt.tight_layout()
    plt.show()
    
    return results_df

# =============================================================================
# MAIN FUNCTION

def main():
    model_path = 'fabric_cnn_model.h5'
    image_dir = 'fabric_images'
    output_csv = 'fabric_predictions.csv'
    
    print("Loading dataset...")
    images, labels = load_dataset('fabric_dataset_labels.csv', image_dir)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print("Building model...")
    model = build_fabric_cnn()
    model.summary()
    
    print("Training model...")
    train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    model.save(model_path)
    
    print("Predicting fabric counts...")
    predictions_df = predict_fabric_counts(model, image_dir, output_csv)
    print(predictions_df.head())

if __name__ == "__main__":
    main()


# In[ ]:




