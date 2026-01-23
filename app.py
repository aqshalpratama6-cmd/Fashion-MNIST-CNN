import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Fashion MNIST Image Classifier",
    page_icon="🎀🪞🩰🧥👜🥾",
    layout="centered"
)

# =====================
# Custom CSS (GAME UI)
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    background-color: #0e1117;
}

.main-title {
    text-align: center;
    color: #00f5ff;
    font-size: 42px;
    font-weight: 800;
    text-shadow: 0 0 15px #00f5ff;
}

.subtitle {
    text-align: center;
    color: #c7d0d9;
    font-size: 15px;
    margin-bottom: 30px;
}

.game-card {
    background: linear-gradient(145deg, #1a1f2e, #111827);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0,245,255,0.15);
    margin-bottom: 25px;
}

.group-card {
    background: linear-gradient(135deg, #2b1055, #7597de);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(117,151,222,0.5);
    text-align: center;
}

.group-title {
    font-size: 22px;
    color: #ffffff;
    margin-bottom: 15px;
    text-shadow: 0 0 10px rgba(255,255,255,0.6);
}

.member {
    color: #e0e7ff;
    font-size: 14px;
    line-height: 1.8;
}

.start-btn button {
    background: linear-gradient(90deg, #00f5ff, #00ffa2) !important;
    color: black !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    height: 3em !important;
    box-shadow: 0 0 15px rgba(0,255,200,0.6);
}

</style>
""", unsafe_allow_html=True)

# =====================
# Load Model
# =====================
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "fashion_mnist_cnn_group9_optimized.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =====================
# Class Labels
# =====================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =====================
# Prediction Function
# =====================
def predict_image(model, image):
    img = image.convert("L").resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    preds = model.predict(img_array)
    return np.argmax(preds), preds[0]

# =====================
# TITLE
# =====================
st.markdown("<div class='main-title'>🧥 FASHION MNIST CLASSIFIER 👗</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload item fashion & let the CNN decide your class</div>", unsafe_allow_html=True)

# =====================
# GAMEPLAY CARD
# =====================
st.markdown("<div class='game-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "🕹️ SELECT ITEM IMAGE",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="🧥 Player Item", use_container_width=True)

    with col2:
        st.markdown("### 🧠 CNN ANALYSIS")

        st.markdown("<div class='start-btn'>", unsafe_allow_html=True)
        start = st.button("▶ START CLASSIFICATION")
        st.markdown("</div>", unsafe_allow_html=True)

        if start:
            with st.spinner("🔄 Processing Level..."):
                pred_class, pred_probs = predict_image(model, image)
                confidence = pred_probs[pred_class] * 100

                st.success(
                    f"🏆 **RESULT:** {class_names[pred_class]}  \n"
                    f"🎯 **ACCURACY:** {confidence:.2f}%"
                )

                st.markdown("### 📊 POWER STATS")
                for i, prob in enumerate(pred_probs):
                    st.progress(
                        float(prob),
                        text=f"{class_names[i]} : {prob:.2f}"
                    )

st.markdown("</div>", unsafe_allow_html=True)

# =====================
# GROUP CARD
# =====================
st.markdown("<div class='group-card'>", unsafe_allow_html=True)
st.markdown("<div class='group-title'>👥 PLAYER TEAM — GROUP 09</div>", unsafe_allow_html=True)

st.markdown("""
<div class='member'>
Aqshal Yanu Pratama <br>
<span style="font-size:12px;">ID: 2802525666</span><br><br>

Andreas Hartono <br>
<span style="font-size:12px;">ID: 2802548493</span><br><br>

Ilham Lazuardy Muhammad Syamlan <br>
<span style="font-size:12px;">ID: 2802519814</span><br><br>

Aulia Aisha Carine <br>
<span style="font-size:12px;">ID: 2802553562</span>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
