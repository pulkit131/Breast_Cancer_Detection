import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("densenet121.h5")
    return model

# Load label encoder
@st.cache_resource
def load_label_encoder():
    with open("label_encoder_pathology.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

model = load_model()
label_encoder = load_label_encoder()

# Streamlit UI
st.title("🩺 Breast Cancer Mass Classification")
st.write("Upload a mammogram image to classify its pathology using the trained DenseNet121 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = load_img(uploaded_file, target_size=(256, 256))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(preds) * 100

    st.success(f"### 🧾 Predicted Pathology: **{predicted_label}**")
    st.write(f"Confidence: {confidence:.2f}%")
