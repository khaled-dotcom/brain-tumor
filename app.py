import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# 1️⃣ اعمل تهيئة للـ client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"
)

# 2️⃣ رفع الصورة من المستخدم
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3️⃣ حول الصورة bytes عشان تقدر تمررها للـ API
    image_bytes = uploaded_file.read()

    # 4️⃣ شغل الـ inference على الصورة
    result = CLIENT.infer(image_bytes, model_id="brain-tumor-s3jcl-qqrkr/2")

    # 5️⃣ اعرض النتايج
    st.write(result)

    # 6️⃣ اعرض الصورة نفسها
    st.image(Image.open(uploaded_file), caption="الصورة المرفوعة")
