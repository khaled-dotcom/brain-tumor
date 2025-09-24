# app.py
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import tempfile

# 1️⃣ تهيئة Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"  # استخدم الـ Private API Key بتاعك
)

st.title("Brain Tumor Detection")
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

    # 5️⃣ تحميل الصورة وتهيئتها للرسم
    image = Image.open(tmp_path)
    draw = ImageDraw.Draw(image)

    # 6️⃣ رسم الصناديق
    for pred in result.get("predictions", []):
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        class_name = pred["class"]
        confidence = pred["confidence"]

        # حساب إحداثيات الصندوق
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        # رسم الصندوق
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left, top - 15), f"{class_name} {confidence:.2f}", fill="red")

    # 7️⃣ عرض الصورة بعد الكشف
    st.subheader("الصورة بعد الكشف عن الأورام")
    st.image(image, use_column_width=True)

    # 8️⃣ عرض البيانات كـ JSON
    st.subheader("تفاصيل الكشف")
    st.json(result)
