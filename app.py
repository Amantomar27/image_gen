from diffusers import DiffusionPipeline
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Load the diffusion model
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe

pipe = load_pipeline()

# Streamlit UI
st.set_page_config(page_title="Image Generator", layout="centered")
st.title("🎨 Text-to-Image Generator")

prompt = st.text_input("📝 Enter your prompt:")
negative_prompt = st.text_input("🚫 Enter negative prompt (optional):", "")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
