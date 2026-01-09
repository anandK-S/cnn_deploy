import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.title("CNN Image Classification")

MODEL_URL = "https://huggingface.co/Anandkumar14065/cnn-model/resolve/main/cnn_model.keras"
MODEL_PATH = "cnn_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ⚠️ Update class names exactly as training
class_names = ["class1", "class2", "class3"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {predicted_class}")
