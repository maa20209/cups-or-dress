import streamlit as st
from huggingface_hub import hf_hub_download
from fastai.vision.all import load_learner

st.set_page_config(page_title="Cups or Dress")

st.title("☕ Cups or Dress?")
st.write("Upload an image and let the model classify it.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="MAA2026/cups-or-dress-model",
        filename="model.pkl"
    )
    return load_learner(model_path)

learn = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image")

    if st.button("Classify"):
        prediction, _, probabilities = learn.predict(uploaded_file)

        st.subheader(f"Prediction: {prediction}")

        for label, probability in zip(learn.dls.vocab, probabilities):
            st.write(f"**{label}**: {probability:.2%}")
