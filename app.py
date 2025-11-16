import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import json

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=7)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Load class labels
with open("labels.json", "r") as f:
    idx_to_class = json.load(f)

# ------------------------------------------
# PREPROCESS FUNCTION
# ------------------------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.title("🩺 Skin Cancer Detection App (EfficientNet-B3)")
st.write("Upload a skin lesion image and the model will classify it into one of the 7 HAM10000 categories.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(image)

    with st.spinner("🔍 Analyzing image..."):
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        classname = idx_to_class[str(predicted.item())]

    st.success(f"### 🧬 Prediction: **{classname.upper()}**")
