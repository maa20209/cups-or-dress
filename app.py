import streamlit as st
import torch
from torchvision.models import resnet18
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms


st.set_page_config(page_title="Cups or Dress")

st.title("☕ Cups or Dress?")
st.write("Upload an image and let the model classify it.")


@st.cache_resource
def load_model():

    # Download the model weights from Hugging Face
    model_path = hf_hub_download(
        repo_id="MAA2026/cups-or-dress-model",
        filename="model_fp16.pth"
    )

    # Create ResNet18
    model = resnet18(weights=None)

    # Re-create the fastai classification head
    model.fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),
        nn.Linear(512, 512, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 2, bias=False)
    )

    # Load the saved weights
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# Same basic image size used during training
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
])


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image")

    if st.button("Classify"):

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        classes = ["cups", "dress"]

        prediction = classes[probabilities.argmax().item()]

        st.subheader(f"Prediction: {prediction}")

        for label, probability in zip(classes, probabilities):
            st.write(f"**{label}**: {probability.item():.2%}")
