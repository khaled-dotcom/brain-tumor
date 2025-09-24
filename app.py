import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# 1️⃣ تهيئة الـ client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"
)

# 2️⃣ رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3️⃣ اعمل ملف مؤقت واحفظ فيه الصورة
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 4️⃣ شغل الـ inference باستخدام مسار الصورة
    result = CLIENT.infer(tmp_path, model_id="brain-tumor-s3jcl-qqrkr/2")

    # 5️⃣ اعرض النتايج
    st.write(result)

    # 6️⃣ اعرض الصورة نفسها
    st.image(Image.open(tmp_path), caption="الصورة المرفوعة")
