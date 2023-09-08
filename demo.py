import streamlit as st
import os
from PIL import Image
import torch

torch.__version__

root_folder = os.path.abspath("")


def load_image():
    uploaded_file = st.file_uploader(label="Pick an image to test")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        with open(
            os.path.join(root_folder, "datasets/dirty2clean/input.jpg"), "wb"
        ) as f:
            f.write(image_data)
        os.system(
            "python test.py --dataroot datasets/dirty2clean --name dirty2clean --model test --no_dropout --gpu_ids -1 --load_size 512 --crop_size 512"
        )
        output_path = os.path.join(
            root_folder, "results/dirty2clean/test_latest/images/input_fake.png"
        )
        output = Image.open(output_path)
        st.image(output, caption="Your beach is now clean")


def main():
    st.title("Image upload demo")
    load_image()


if __name__ == "__main__":
    main()
