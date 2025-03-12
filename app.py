import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set the page configuration
st.set_page_config(layout="wide", page_title="Caster Disease Classification")

# Path to the saved model
MODEL_PATH = "cnn_cucumber_classifier.h5"

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Define class labels
CLASS_NAMES = ["Healthy", "Unhealthy"]

st.title("Caster Disease Classification Dashboard")

# Creating two columns for a split-screen layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # ✅ Preprocess the image (Resize to 64x64 to match model input)
        img = image.resize((64, 64))  # Change from (224, 224) to (64, 64)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # ✅ Ensure the correct shape before prediction
        img_array = img_array.astype(np.float32)  

        # Make prediction
        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

with col2:
    st.subheader("Caster Disease Information")
    st.markdown(
        """
        **Caster Disease Overview:**
        Caster plants can be affected by various diseases that impact growth and yield.
        
        **Common Diseases:**
        - **Fungal Infections** – Caused by *Fusarium wilt*, *Alternaria*, etc.
        - **Bacterial Wilt** – Affects leaf structure and plant health.
        - **Viral Infections** – Spread through insects and poor soil health.
        
        **Treatment & Prevention:**
        - Use disease-resistant varieties.
        - Maintain proper soil moisture and drainage.
        - Apply fungicides like *Mancozeb 75% WP* or *Copper Oxychloride 50% WP*.
        - Rotate crops to reduce soil-borne diseases.
        """
    )

st.markdown("---")
st.write("Developed by Anurag using Streamlit ❤️")
