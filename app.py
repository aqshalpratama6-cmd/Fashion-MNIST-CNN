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
# UI Streamlit (Enhanced)
# =====================

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
        👕 Fashion MNIST Classifier 👟
    </h1>
    <p style='text-align: center; font-size:16px;'>
        Upload gambar pakaian dan lihat hasil klasifikasi dari model CNN
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload image (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.markdown("### 🖼️ Preview Image")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Classification Result")

        classify_btn = st.button(
            "✨ Classify Image",
            use_container_width=True
        )

        if classify_btn:
            with st.spinner("🧠 Model is thinking..."):
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
