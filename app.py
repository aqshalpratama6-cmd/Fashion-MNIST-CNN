import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fashion_mnist_model.h5')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(model, image):
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    preds = model.predict(img_array)
    return np.argmax(preds), preds[0]

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title('Fashion MNIST Image Classifier')
st.write('Upload a fashion item image.')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption='Uploaded Image', width=120)
    with col2:
        st.write('')
        if st.button('Classify Image', use_container_width=True):
            with st.spinner('Classifying...'):
                pred_class, pred_probs = predict_image(model, image)
                st.success(f'Prediction: **{class_names[pred_class]}**')
        else:
            st.info('Click the button to classify the image.')