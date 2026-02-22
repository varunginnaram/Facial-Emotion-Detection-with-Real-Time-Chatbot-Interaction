import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Set paths to the dataset and model
train_dir = "D:/College/Sem 5/DeepLearning/Project/train"  # Path to the train folder
test_dir = "D:/College/Sem 5/DeepLearning/Project/test"    # Path to the test folder
model_path = "enhanced_emotion_recognition_cnn_model.h5"    # Path to the saved model

# Define image size and batch size
image_size = (48, 48)  # Typical size for FER2013 images
batch_size = 64

# Data preprocessing
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale

# Test data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ensure order for evaluation
)

# Check if the model already exists
if os.path.exists(model_path):
    # Load the saved model
    print("Model is already trained and saved. Loading the model...")
    model = tf.keras.models.load_model(model_path)

else:
    # Model training setup
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values
        validation_split=0.2  # Use 20% of training data for validation
    )

    # Training data
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Use training subset
    )

    # Validation data
    val_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Use validation subset
    )

    # Build the CNN model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print("Training the model...")
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    epochs = 50
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[lr_scheduler]
    )

    # Save the model
    model.save(model_path)
    print("Model is trained and saved.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Get predictions and ground truth for evaluation
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification report and confusion matrix
target_names = list(test_data.class_indices.keys())
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=target_names))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
