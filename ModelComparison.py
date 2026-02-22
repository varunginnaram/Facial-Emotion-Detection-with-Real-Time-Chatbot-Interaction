import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from deepface import DeepFace
import os


test_dir = "D:/College/Sem 5/DeepLearning/Project/test" 
model_path = "enhanced_emotion_recognition_cnn_model.h5"

image_size = (48, 48)
batch_size = 64


test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ensure order for evaluation
)


cnn_model = tf.keras.models.load_model(model_path)


cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test_data)
print(f"Custom CNN Model - Test Loss: {cnn_test_loss:.4f}")
print(f"Custom CNN Model - Test Accuracy: {cnn_test_accuracy:.4f}")


cnn_predictions = cnn_model.predict(test_data)
cnn_y_pred = np.argmax(cnn_predictions, axis=1)
cnn_y_true = test_data.classes


cnn_target_names = list(test_data.class_indices.keys())
print("\nCustom CNN Model - Classification Report:\n")
print(classification_report(cnn_y_true, cnn_y_pred, target_names=cnn_target_names))

print("\nCustom CNN Model - Confusion Matrix:\n")
print(confusion_matrix(cnn_y_true, cnn_y_pred))

# Define emotion categories for DeepFace
emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize variables for ground truth and predictions
df_y_true = []
df_y_pred = []

# Loop through each emotion folder for DeepFace
for i, emotion in enumerate(emotion_categories):
    emotion_folder = os.path.join(test_dir, emotion)
    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        try:
            # Predict emotion using DeepFace
            result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):  
                result = result[0]
            
            # Get the dominant emotion
            predicted_emotion = result['dominant_emotion']
            
            # Append ground truth and prediction
            df_y_true.append(i)
            df_y_pred.append(emotion_categories.index(predicted_emotion))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

print("\nDeepFace Model - Classification Report:\n")
print(classification_report(df_y_true, df_y_pred, target_names=emotion_categories))

print("\nDeepFace Model - Confusion Matrix:\n")
print(confusion_matrix(df_y_true, df_y_pred))

print("\n--- Comparison of Custom CNN Model and DeepFace ---\n")
print(f"Custom CNN Model Accuracy: {cnn_test_accuracy:.4f}")
print(f"DeepFace Model Accuracy: {np.mean(np.array(df_y_true) == np.array(df_y_pred)):.4f}")

# Custom CNN vs DeepFace - Conclusions
if cnn_test_accuracy > np.mean(np.array(df_y_true) == np.array(df_y_pred)):
    print("Conclusion: The Custom CNN model performs better based on accuracy.")
else:
    print("Conclusion: The DeepFace model performs better based on accuracy.")
