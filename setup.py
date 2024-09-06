"""Setup script."""

from setuptools import setup

setup(
    name="vtamp",
    version="0.1.0",
    packages=["vtamp"],
    include_package_data=True,
    install_requires=[
        "pybullet==3.2.6",
        "numpy==1.26.4",
        "openai==1.16.2",
        "python-dotenv==1.0.1",
        "imageio==2.34.0",
        "easydict==1.12",
        "opencv-python==4.9.0.80",
        "Pillow==9.5.0",
        "jax==0.4.25",
        "jaxlib==0.4.25",
        "clip",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "matplotlib==3.8.3",
        "flax==0.8.1",
        "docformatter==1.7.5",
        "black==24.2.0",
        "isort==5.13.2",
        "tensorflow==2.16.1",
        "orbax-checkpoint==0.5.3",
        "optax==0.2.1",
        "shapely==2.0.3",
        "clip @ git+https://github.com/openai/CLIP.git#egg=clip",
        "hydra_colorlog==1.2.0",
        "pystache==0.6.5",
    ],
)
