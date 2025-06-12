import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ページ設定（最初に書くこと）
st.set_page_config(page_title="♻️ ゴミ分別AI (Paderborn)", page_icon="🗑️")

# モデル読み込み
model = load_model("model/classifier_kaggle_paderborn.h5")

# クラスラベル（processed_datasetのサブフォルダ順に合わせる）
class_labels = ['biowaste', 'glass', 'paper', 'residual', 'wertstoff']

# 画像を予測する関数
def predict(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    return class_labels[class_idx], confidence

# Streamlit UI
st.title("♻️ Paderbornごみ分別AI")
st.write("アップロードした画像から、どのごみ箱に入れるべきかを予測します。")

uploaded_file = st.file_uploader("画像をアップロードしてください（例: ペットボトル、紙、残飯など）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="アップロード画像", use_column_width=True)

    with st.spinner("分類中..."):
        label, confidence = predict(uploaded_file)
        st.success(f"**予測: `{label}`（信頼度: {confidence:.2%}）**")

    st.markdown("---")
    st.caption("モデル: MobileNetV2 + 転移学習（KaggleデータベースをPaderborn分類に変換）")
