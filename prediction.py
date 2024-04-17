import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
saved_model_path = r"C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\inception_model.h5"

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Load the model without compiling
loaded_model = load_model(saved_model_path, compile=False)

# Now, compile the model manually
loaded_model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (224, 224))
    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    expanded_image = np.expand_dims(normalized_image, axis=0)
    return expanded_image

# Function to predict the behavior of the image
def predict_behavior(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make prediction using the loaded model
    predictions = loaded_model.predict(preprocessed_image)
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])
    # Map the predicted class index to the corresponding behavior
    behaviors = {0: 'Other', 1: 'Safe Driving', 2: 'Talking Phone', 3: 'Texting Phone', 4: 'Turning'}
    predicted_behavior = behaviors[predicted_class_index]
    return predicted_behavior

# Path of the image to be predicted
image_path = r"C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\img_928.jpg"

# Predict the behavior of the image
predicted_behavior = predict_behavior(image_path)
print("Predicted behavior:", predicted_behavior)
