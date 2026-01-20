import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fashion_mnist_cnn_optimized.h5")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Fashion MNIST CNN Classifier")

uploaded_file = st.file_uploader(
    "Upload gambar pakaian (jpg/png)",
    type=["jpg","png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    st.image(img, caption="Gambar Input")

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    st.subheader("Hasil Prediksi")
    st.write("Kategori:", class_names[class_idx])
    st.write("Confidence:", f"{np.max(prediction)*100:.2f}%")
