import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Load the saved model
saved_model_path = r"C:\Users\anand\OneDrive\Documents\Driver_Behaviour_Detection\inception_model.h5"

# Load the model without compiling
loaded_model = load_model(saved_model_path, compile=False)

# Now, compile the model manually
loaded_model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (224, 224))
    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    expanded_image = np.expand_dims(normalized_image, axis=0)
    return expanded_image

# Function to predict the behavior of the image
def predict_behavior(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make prediction using the loaded model
    predictions = loaded_model.predict(preprocessed_image)
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])
    # Map the predicted class index to the corresponding behavior
    behaviors = {0: 'Doing Other activites except talking,texting,turing', 1: 'Safely Driving', 2: 'Talking Phone', 3: 'Texting Phone', 4: 'Turning'}
    predicted_behavior = behaviors[predicted_class_index]
    return predicted_behavior

# Main Streamlit app
def main():
    st.title("Driver Behavior Detection")

    uploaded_file = st.file_uploader("Choose an image of driving...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting the image...")

        predicted_behavior = predict_behavior(image)

        st.write("The driver's behaviour while driving is:", predicted_behavior)

if __name__ == "__main__":
    main()