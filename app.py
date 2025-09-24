import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# Initialize Roboflow Inference client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"  # Replace with your private API key
)

st.title("Brain Tumor Detection using Roboflow API")
st.write("Upload an image and the model will detect brain tumors.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run inference
    with st.spinner("Running inference..."):
        result = CLIENT.infer(uploaded_file, model_id="brain-tumor-s3jcl-qqrkr/2")  # your model ID

    st.success("Inference done!")
    st.write(result)  # Display raw JSON results
