import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas


if 'model' not in st.session_state:
    st.session_state.model = tf.keras.models.load_model('mnist_cnn_model.h5', compile=False)


def preprocess_image(img):
    img = img.resize((28, 28))  
    img = img.convert('L')  
    img = ImageOps.invert(img)  
    img_array = np.array(img).astype('float32') / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1)  
    return img_array


def predict_digit(img):
    img_array = preprocess_image(img)
    prediction = st.session_state.model.predict(img_array)
    return np.argmax(prediction)

# Streamlit app
st.title('Draw a Digit')
st.write('Use the canvas below to draw a digit and click Predict.')


canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)


if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = None

if canvas_result.image_data is not None:
    st.session_state.canvas_data = canvas_result.image_data

if st.button("Predict") and st.session_state.canvas_data is not None:
    image = Image.fromarray(st.session_state.canvas_data.astype('uint8'), 'RGBA').convert('L')
    digit = predict_digit(image)
    st.success(f"Predicted Digit: {digit}")

#This part actually not needed,just clicking the delete button at the bottom of the canvas to clear the canvas
if st.button("Clear"):
    st.session_state.canvas_data = None  # Clear stored data
    st.experimental_rerun()
