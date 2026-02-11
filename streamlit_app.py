import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="ResNet50 Classifier", page_icon="🔬", layout="centered")
st.title("🔬 ResNet50 Classifier")
st.write("Upload an image to classify its condition (Critical, Low, or Normal).")


# 2. Load Model
@st.cache_resource
def load_model():
    # Define architecture (EfficientNetV2-S)
    model = models.efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 3),
    )

    # Load weights (FP16 weights in file)
    model_path = "efficientnet_v2_compressed.pth"
    state_dict = torch.load(model_path, map_location="cpu")

    # Cast weights back to float32 for CPU inference stability
    float32_state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    model.load_state_dict(float32_state_dict)
    model.eval()
    return model


try:
    model = load_model()
    class_names = ["critical", "low", "normal"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 4. UI components
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "tiff", "tif"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf = probs[0][preds[0]].item()
                label = class_names[preds[0]]

            # Show result
            st.success(f"**Prediction:** {label.upper()}")
            st.info(f"**Confidence:** {conf*100:.2f}%")

            # Confidence bar chart
            st.bar_chart(
                {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
            )

st.markdown("---")
st.caption("Model compressed using FP16 weight quantization for efficient deployment.")
