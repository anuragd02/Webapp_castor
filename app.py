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

import pandas as pd

with col2:
    st.subheader("Castor – Botrytis Grey Mold (BGM)")

    st.markdown("### Crop–Disease–Pathogen–Fungicide Overview")

    overview_df = pd.DataFrame({
        "Crop": ["Castor\n(Ricinus communis L.)"],
        "Disease": ["Botrytis Grey Mold\n(BGM)"],
        "Causal Pathogen": ["Botrytis cinerea\nPers. ex Fr."],
        "Fungicide Used": ["Propiconazole\n25 EC"],
        "Mode of Action": [
            "Systemic triazole fungicide\n"
            "Inhibits ergosterol biosynthesis (DMI)"
        ]
    })

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Key Symptoms of Botrytis Grey Mold (BGM)")
    st.markdown(
        """
        - Water-soaked lesions on spikes and floral parts  
        - Grey to brown discoloration of spikelets  
        - Soft rotting of flowers and capsules under humid conditions  
        - Grey, fuzzy fungal growth visible during high humidity  
        - Premature drying and shrivelling of spike tissues  
        - Flower drop and poor capsule setting  
        - Musty odour from infected spikes during severe infection  
        - Rapid disease spread during cloudy weather and intermittent rainfall  
        - Entire spike rot under prolonged or severe infection  
        - Reduced seed size and inferior seed quality
        """
    )

    st.markdown("### Propiconazole 25 EC – Dose Optimization Strategy")

    dose_df = pd.DataFrame({
        "Goal": [
            "Maximum yield",
            "Cost-effective + high yield",
            "Most stable across environments",
            "Low disease pressure",
            "Not recommended"
        ],
        "Best Treatment": [
            "2.5 ml/L",
            "2.0 ml/L",
            "1.5 ml/L",
            "1.0 ml/L",
            "0.5 ml/L"
        ],
        "Why": [
            "Highest yield and maximum benefit–cost ratio",
            "Yield close to maximum with reduced input cost",
            "Best AMMI stability and consistent performance",
            "Adequate disease control with minimal cost",
            "Insufficient disease control and low economic returns"
        ]
    })

    st.dataframe(
        dose_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### DSS-Based Advisory Recommendation")
    st.markdown(
        """
        - **High BGM risk** → Propiconazole 25 EC @ **2.5 ml/L**  
        - **Moderate BGM risk** → Propiconazole 25 EC @ **2.0 ml/L**  
        - **Variable / unstable environment** → Propiconazole 25 EC @ **1.5 ml/L**  
        - **Early or mild infection** → Propiconazole 25 EC @ **1.0 ml/L**  
        - **Avoid** → **0.5 ml/L** due to poor disease suppression
        """
    )

st.markdown("---")
st.markdown(
    """
    <style>
    .developed-by {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .person {
        font-size: 16px;
        margin-bottom: 2px;
    }
    </style>
    <div class="developed-by">Developed by</div>
    <div class="person"><b>Anurag Dhole</b> - Researcher at MIT, Manipal</div>
    <div class="person"><b>Dr. Jadesha G</b> - Assistant Professor at GKVK, UAS, Bangalore</div>
    <div class="person"><b>Dr. Deepak D.</b> - Professor at MIT, Manipal</div>
    """,
    unsafe_allow_html=True
)

