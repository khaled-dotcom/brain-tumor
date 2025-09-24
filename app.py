# app.py
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# 1️⃣ تهيئة الـ Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"  # استخدم الـ Private API Key بتاعك
)

st.title("Brain Tumor Detection with Roboflow")
st.write("ارفع صورة لفحصها بالكشف عن الأورام")

# 2️⃣ رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3️⃣ حفظ الصورة مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 4️⃣ تشغيل الـ inference
    result = CLIENT.infer(tmp_path, model_id="brain-tumor-s3jcl-qqrkr/2")

    # 5️⃣ عرض النتائج
    st.subheader("نتيجة التحليل")
    st.json(result)  # ده هيعرضلك كل تفاصيل النتيجة (مثل boxes, classes, confidence)

    # 6️⃣ عرض الصورة الأصلية
    st.subheader("الصورة الأصلية")
    st.image(Image.open(tmp_path), caption="الصورة المرفوعة", use_column_width=True)
