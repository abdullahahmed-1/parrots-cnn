import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Bird Classifier",
    page_icon="🐦",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    model_path = 'bird_model.h5'
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict
def predict(image, model):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Preprocess (MobileNetV2 specific: [-1, 1])
    # The training used tf.keras.applications.mobilenet_v2.preprocess_input
    # which expects inputs in [0, 255] then scales them.
    # So we just pass the array to the model input if the model has the preprocessing layer built-in?
    # In my notebook, I included `preprocess_input` as a layer *before* the base model.
    # So I just need to provide the image array.
    
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return np.argmax(score), np.max(score)

# Header
st.title("🐦 Bird Species Classifier")
st.write("Upload an image of a bird to identify its species.")

# Load Model
model = load_model()

if model is None:
    st.error("Model file `bird_model.h5` not found. Please train the model using the notebook and place the .h5 file in the same directory.")
else:
    # File Uploader
    file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("")
        st.write("Classifying...")
        
        # Hardcoded classes based on the folder contents from earlier tool output
        # Verify these match the notebook's sorted order (default for image_dataset_from_directory)
        # Folders: amazon green parrot.jpg, gray parrot.jpg, macaw.jpg, white parrot.jpg
        # Sorted: ['amazon green parrot.jpg', 'gray parrot.jpg', 'macaw.jpg', 'white parrot.jpg']
        class_names = ['Amazon Green Parrot', 'Gray Parrot', 'Macaw', 'White Parrot']
        original_folder_names = ['amazon green parrot.jpg', 'gray parrot.jpg', 'macaw.jpg', 'white parrot.jpg']
        
        # Make Prediction
        try:
            class_idx, confidence = predict(image, model)
            
            st.success(f"Prediction: **{class_names[class_idx]}**")
            st.info(f"Confidence: {confidence:.2%}")
            
            # Progress bar for confidence
            st.progress(float(confidence))
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
