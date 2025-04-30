from diffusers import DiffusionPipeline  # Import the DiffusionPipeline class from the diffusers library
import torch  # Import the torch library for tensor operations

# Initialize the base pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Specify the model checkpoint to load for the base pipeline
    torch_dtype=torch.float16,  # Set the data type for tensors to float16, which is half-precision floating point
    use_safetensors=True,  # Enable safe tensors, which is typically used for numerical stability (Safe tensors are designed to improve numerical stability during model operations. By using safe tensors, the model can potentially reduce the risk of numerical errors and improve the overall robustness of the computations.)
    variant="fp16"  # Specify the variant as fp16, indicating the model is optimized for float16 operations (The variant parameter is set to "fp16", indicating that the model is optimized for operations using float16 precision. Float16, also known as half-precision, reduces the memory footprint and computational load compared to full precision (float32). This variant is particularly useful for leveraging hardware accelerators like GPUs that perform well with half-precision arithmetic, leading to faster computation and reduced memory usage.)
)

pipe.to("cuda")  # Move the pipeline to GPU (CUDA) for faster computation if CUDA is available

def generate_image(prompt,negative_prompt):
  image=pipe(prompt=prompt,negative_prompt=negative_prompt).images[0]
  import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()

import streamlit as st
st.title("Image Generator")
query=st.text_input("enter your text")
submit=st.text_input(query)
if submit:
  generate_image(query,submit)
  