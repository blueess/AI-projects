import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("traffic_sign_model.keras")

# Class names
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 
    'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 
    'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Click 'Browse files' to upload a traffic sign image (jpg, jpeg, png).
2. Wait for the prediction to appear below the image.
3. The predicted traffic sign class will be shown clearly.
""")

# Add accuracy disclaimer
st.sidebar.warning("Note: The model's predictions may not be highly accurate. Please use results with caution.")

st.title("🚦 Traffic Sign Recognition")
st.markdown("""
<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;'>
<b style='color:black;'>Upload an image of a traffic sign below.<br>
The AI model will predict the type of sign.</b>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image', use_column_width=True)
    st.write("")

    with st.spinner('Analyzing image and predicting...'):
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: ")
    st.markdown(f"<h2 style='color:#0072C6;'>{predicted_class}</h2>", unsafe_allow_html=True)