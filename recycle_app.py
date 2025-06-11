import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="♻️ ゴミ分別AI", page_icon="🗑️")
# ラベル一覧
LABELS = ["biowaste", "plastic", "glass", "paper", "pfand", "residual"]

# モデル読み込み
@st.cache_resource
def load_classifier():
    return load_model("model/classifier.h5")

model = load_classifier()

# UI
st.title("♻️ Paderborn ごみ分別アプリ")
st.caption("ゴミの写真から、分別カテゴリをAIが予測します。")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    resized = image.resize((224, 224))
    image_array = np.array(resized)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)
    predicted_label = LABELS[predicted_index]

    st.success(f"推定カテゴリ：**{predicted_label}**")

    explanations = {
        "biowaste": "生ゴミ（果物の皮、茶殻など）",
        "plastic": "プラスチック包装（カップ、袋など）",
        "glass": "ガラス瓶（透明・茶・緑）",
        "paper": "紙類（新聞、段ボールなど）",
        "pfand": "♻️ Pfand付き容器（リサイクル可ペット・缶）",
        "residual": "その他（汚れたティッシュ、歯ブラシ等）"
    }

    st.markdown(f"**説明**：{explanations[predicted_label]}")
