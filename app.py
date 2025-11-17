import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import torch.nn.functional as F

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")

st.markdown(
    "<h1 style='text-align: center;'>🩺 Skin Cancer Detection (EfficientNet-B0)</h1>",
    unsafe_allow_html=True
)

st.write("### Upload a skin lesion image for classification.")


# ----------------------------------------------------
# LOAD MODEL (Guaranteed Working)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


# ----------------------------------------------------
# LOAD LABELS
# ----------------------------------------------------
with open("labels.json", "r") as f:
    idx_to_class = json.load(f)


# ----------------------------------------------------
# PREPROCESS FUNCTION
# ----------------------------------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(img)

    # ------------------------------------------------
    # PREDICTION
    # ------------------------------------------------
    with st.spinner("🔍 Classifying..."):
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).detach().numpy()[0]

        pred_idx = int(torch.argmax(outputs, 1))
        pred_label = idx_to_class[str(pred_idx)]
        confidence = probs[pred_idx] * 100

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------
    st.success(f"### 🧬 Prediction: **{pred_label.upper()}**")
    st.write(f"### 🔢 Confidence: **{confidence:.2f}%**")
    st.progress(confidence / 100)

    # Probability Breakdown
    st.subheader("📊 Probability Breakdown")
    for i, p in enumerate(probs):
        st.write(f"**{idx_to_class[str(i)].upper()}** → {p * 100:.2f}%")
