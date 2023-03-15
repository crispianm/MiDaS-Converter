# MiDaS-Converter

This is a script that will create a duplicate of a given directory, containing depth estimates for every image contained in the given directory and its subfolders. 

## Usage
``python run.py <input_folder> <output_folder>``

For example, given the folder ``data`` in this repository, containing images and a subfolder of images, we can create a depth estimate folder ``output`` by running ``python run.py "./data" "./output"`` in the repo directory.

By default, the image converter will use the highest quality MiDaS model, DPT_BEiT_L_512, taken from the [MiDaS GitHub repo](https://github.com/isl-org/MiDaS).

## Live Depth Estimation
Running ``midas_livecv.py`` will compute relative depth live, but performs best with a dedicated GPU. By default, this is set to use the fastest MiDaS model, DPT_LeViT_224.
