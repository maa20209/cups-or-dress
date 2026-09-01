import streamlit as st
import torch

from fastai.vision.all import *
from fastai.vision.learner import create_vision_model
from huggingface_hub import hf_hub_download
from PIL import Image


st.set_page_config(page_title="Cups or Dress")

st.title("☕ Cups or Dress?")
st.write("Upload an image and let the model classify it.")


@st.cache_resource
def load_model():

    # Download the trained weights from Hugging Face
    model_path = hf_hub_download(
        repo_id="MAA2026/cups-or-dress-model",
        filename="model_fp16.pth"
    )

    # Re-create the SAME type of model used by fastai vision_learner
    model = create_vision_model(
        resnet18,
        n_out=2,
        pretrained=False,
        ps=0.5
    )

    # Load our saved weights
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# Same image size used during training
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("Classify"):

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        classes = ["cups", "dress"]

        prediction_index = probabilities.argmax().item()
        prediction = classes[prediction_index]

        st.subheader(f"Prediction: {prediction}")

        for label, probability in zip(classes, probabilities):
            st.write(
                f"**{label}**: {probability.item():.2%}"
            )
