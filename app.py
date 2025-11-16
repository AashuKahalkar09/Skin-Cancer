import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import json
import torch.nn.functional as F

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>🩺 Skin Cancer Detection (EfficientNet-B0)</h1>", 
            unsafe_allow_html=True)
st.write("### Upload a skin lesion image to detect possible cancer type.")

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, 7)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------------------
# LOAD LABELS
# -----------------------------------------
with open("labels.json", "r") as f:
    idx_to_class = json.load(f)

# -----------------------------------------
# PREPROCESS
# -----------------------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.header("Navigation")
st.sidebar.info("Use this tool to classify skin lesion images using a trained EfficientNet-B0 model.")
st.sidebar.write("**Model:** EfficientNet-B0")  
st.sidebar.write("**Classes:** 7 (HAM10000 dataset)")  

# -----------------------------------------
# FILE UPLOAD
# -----------------------------------------
uploaded_file = st.file_uploader("📤 Upload Image (JPG/PNG)", 
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", 
             use_container_width=True)

    img_tensor = preprocess_image(image)

    with st.spinner("🔍 Predicting..."):
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1).detach().numpy()[0]
        predicted_class = int(torch.argmax(outputs, 1))
        classname = idx_to_class[str(predicted_class)]
        confidence = probabilities[predicted_class] * 100

    # -----------------------------------------
    # RESULT DISPLAY
    # -----------------------------------------
    st.success(f"## 🧬 Prediction: **{classname.upper()}**")
    st.write(f"### 🔢 Confidence: **{confidence:.2f}%**")

    # CONFIDENCE BAR
    st.progress(float(confidence / 100))

    # Table of all probabilities
    st.subheader("📊 Probability Breakdown")
    for i, prob in enumerate(probabilities):
        st.write(f"**{idx_to_class[str(i)].upper()}** → {prob * 100:.2f}%")
