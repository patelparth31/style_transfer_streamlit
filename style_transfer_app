import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pre-trained style transfer model
model = tf.keras.applications.EfficientNetB0(weights='imagenet')

# Function to perform style transfer
def style_transfer(primary_image, target_image):
    # Preprocess the images
    primary_image = tf.keras.applications.efficientnet.preprocess_input(primary_image)
    target_image = tf.keras.applications.efficientnet.preprocess_input(target_image)

    # Perform style transfer
    primary_feature = model(primary_image)
    target_feature = model(target_image)

    # Combine features
    combined_feature = primary_feature + target_feature

    # Generate the stylized image
    stylized_image = model.layers[0](combined_feature)

    return stylized_image

# Streamlit UI
st.title("Style Transfer App")

# Upload images
primary_image = st.file_uploader("Upload Primary Photo", type=["jpg", "jpeg", "png"])
target_image = st.file_uploader("Upload Target Photo", type=["jpg", "jpeg", "png"])

# Style transfer button
if st.button("Perform Style Transfer"):
    if primary_image is not None and target_image is not None:
        # Read and preprocess images
        primary_image = Image.open(primary_image).convert("RGB")
        target_image = Image.open(target_image).convert("RGB")

        primary_image = np.array(primary_image)
        target_image = np.array(target_image)

        # Perform style transfer
        stylized_image = style_transfer(primary_image, target_image)

        # Display the images
        st.image([primary_image, target_image, stylized_image.numpy()], caption=['Primary', 'Target', 'Stylized'], width=300)

    else:
        st.warning("Please upload both primary and target photos.")
