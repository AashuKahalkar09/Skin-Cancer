import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")

st.markdown(
    "<h1 style='text-align: center;'>🩺 Skin Cancer Detection (YOLO + EfficientNet-B0)</h1>",
    unsafe_allow_html=True
)

st.write("### Upload a skin lesion image for detection & classification.")


# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
@st.cache_resource
def load_models():
    # Load EfficientNet model
    eff_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    eff_model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    eff_model.eval()

    # Load YOLOv11n model
    yolo_model = YOLO("yolo11n.pt")  # You provided this file

    return eff_model, yolo_model


eff_model, yolo_model = load_models()


# ----------------------------------------------------
# LOAD LABELS
# ----------------------------------------------------
with open("labels.json", "r") as f:
    idx_to_class = json.load(f)


# ----------------------------------------------------
# PREPROCESS FOR EFFICIENTNET
# ----------------------------------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


# ----------------------------------------------------
# UPLOAD IMAGE
# ----------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # YOLO DETECTION
    # -------------------------------
    st.subheader("🔍 YOLO Lesion Detection")
    results = yolo_model(img)

    # Convert result to image with bounding boxes
    result_img = results[0].plot()  # YOLO auto draws boxes
    st.image(result_img, caption="Detected Lesions", use_container_width=True)

    # If YOLO found no boxes → direct classification
    if len(results[0].boxes) == 0:
        st.warning("⚠ No lesion detected by YOLO. Running classification on full image.")
        lesion_crop = img
    else:
        # Take first detected lesion
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

        # Crop detected lesion
        np_img = np.array(img)
        lesion_crop = Image.fromarray(np_img[y1:y2, x1:x2])

    st.subheader("🩹 Cropped Lesion for Classification")
    st.image(lesion_crop, use_container_width=True)

    # -------------------------------
    # CLASSIFICATION (EfficientNet)
    # -------------------------------
    st.subheader("🧬 EfficientNet Classification")

    lesion_tensor = preprocess_image(lesion_crop)

    with st.spinner("Classifying..."):
        outputs = eff_model(lesion_tensor)
        probs = F.softmax(outputs, dim=1).detach().numpy()[0]

        pred_idx = int(torch.argmax(outputs, 1))
        pred_label = idx_to_class[str(pred_idx)]
        confidence = probs[pred_idx] * 100

    st.success(f"### 🧬 Predicted: **{pred_label.upper()}**")
    st.write(f"### 🔢 Confidence: **{confidence:.2f}%**")
    st.progress(confidence / 100)

    # Breakdown table
    st.subheader("📊 Probability Breakdown")
    for i, p in enumerate(probs):
        st.write(f"**{idx_to_class[str(i)].upper()}** → {p * 100:.2f}%")
