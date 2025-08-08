import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("skin_cancer_model.h5")
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMAGE_SIZE = 96

st.title("🧴 Skin Cancer Classifier")
st.write("Upload a skin lesion image to predict the cancer class.")

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_label.upper()}**")
