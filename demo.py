import streamlit as st
import os
from PIL import Image
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import shutil
import math


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


root_folder = os.path.abspath("")


def test():
    opt = Namespace(
        aspect_ratio=1.0,
        batch_size=1,
        checkpoints_dir="./checkpoints",
        crop_size=512,
        dataroot="datasets/dirty2clean",
        dataset_mode="single",
        direction="AtoB",
        display_id=-1,
        display_winsize=256,
        epoch="latest",
        eval=False,
        gpu_ids=[],
        init_gain=0.02,
        init_type="normal",
        input_nc=3,
        isTrain=False,
        load_iter=0,
        load_size=512,
        max_dataset_size=math.inf,
        model="test",
        model_suffix="",
        n_layers_D=3,
        name="dirty2clean",
        ndf=64,
        netD="basic",
        netG="resnet_9blocks",
        ngf=64,
        no_dropout=True,
        no_flip=True,
        norm="instance",
        num_test=50,
        num_threads=0,
        output_nc=3,
        phase="test",
        preprocess="resize_and_crop",
        results_dir="./results/",
        serial_batches=True,
        suffix="",
        use_wandb=False,
        verbose=False,
        wandb_project_name="CycleGAN-and-pix2pix",
    )
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print("processing (%04d)-th image... %s" % (i, img_path))
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
        )
    webpage.save()  # save the HTML


def cleaning_degree_to_iterations(cleaning_degree):
    degrees_map = {"Light": 1, "Moderate": 3, "Extensive": 5}
    if degrees_map.get(cleaning_degree) == None:
        return 1
    return degrees_map.get(cleaning_degree)


def load_image():
    uploaded_file = st.file_uploader(label="Upload a Shore you would like to Restore")
    cleaning_degree = st.selectbox(
        "Select Cleaning Intensity:", ("Light", "Moderate", "Extensive")
    )
    no_of_iter = cleaning_degree_to_iterations(cleaning_degree)
    if uploaded_file is not None:
        with st.spinner("Cleaning operation is in progress. Please wait..."):
            image_data = uploaded_file.getvalue()
            st.image(image_data)
            input_path = os.path.join(root_folder, "datasets/dirty2clean/input.jpg")
            with open(input_path, "wb") as f:
                f.write(image_data)
            test()
            output_path = os.path.join(
                root_folder, "results/dirty2clean/test_latest/images/input_fake.png"
            )
            if no_of_iter > 1:
                for i in range(no_of_iter - 1):
                    shutil.copyfile(output_path, input_path)
                    test()
            output = Image.open(output_path)
        st.success("Done!")
        st.image(output, caption="Your Shore is now Restored")


def main():
    st.title("ShoreRestore")
    load_image()


if __name__ == "__main__":
    main()
