import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="centered"
)

# =====================
# Load Model (AMAN)
# =====================
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "fashion_mnist_cnn_group9_optimized.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =====================
# Fungsi Prediksi
# =====================
def predict_image(model, image):
    img = image.convert("L").resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    preds = model.predict(img_array)
    return np.argmax(preds), preds[0]

# =====================
# Label Kelas
# =====================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =====================
# Header UI
# =====================
st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>
        👕 Fashion MNIST Image Classifier 👟
    </h1>
    <p style='text-align:center; font-size:16px;'>
        CNN-based image classification for fashion items
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================
# Upload Section
# =====================
uploaded_file = st.file_uploader(
    "📤 Upload image (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.markdown("### 🖼️ Image Preview")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Classification Result")

        if st.button("✨ Classify Image", use_container_width=True):
            with st.spinner("🧠 Model is analyzing the image..."):
                pred_class, pred_probs = predict_image(model, image)
                confidence = pred_probs[pred_class] * 100

                st.success(
                    f"**Prediction:** {class_names[pred_class]}  \n"
                    f"**Confidence:** {confidence:.2f}%"
                )

                st.markdown("### 📊 Class Probabilities")
                for i, prob in enumerate(pred_probs):
                    st.progress(
                        float(prob),
                        text=f"{class_names[i]}: {prob:.2f}"
                    )
        else:
            st.info("👆 Klik tombol **Classify Image** untuk memulai.")

# =====================
# Group Information Card
# =====================
st.markdown("---")

st.markdown(
    """
    <div style="
        background-color:#F8F9FA;
        padding:20px;
        border-radius:12px;
        box-shadow:0 4px 10px rgba(0,0,0,0.08);
        max-width:600px;
        margin:auto;
        text-align:center;
    ">
        <h3 style="color:#FF4B4B; margin-bottom:10px;">
            👥 Group 09
        </h3>
        <p style="font-size:15px; line-height:1.6; color:#333;">
            <b>Aqshal Yanu Pratama</b> <span style="color:#777;">(2802525666)</span><br>
            <b>Andreas Hartono</b> <span style="color:#777;">(2802548493)</span><br>
            <b>Ilham Lazuardy Muhammad Syamlan</b> <span style="color:#777;">(2802519814)</span><br>
            <b>Aulia Aisha Carine</b> <span style="color:#777;">(2802553562)</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
