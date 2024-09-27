from setuptools import setup, find_packages

setup(
    name="torchastic",
    version="0.1.1",
    author="Lodestone",
    author_email="lodestone.rock@gmail.com",
    description="Stochastic bfloat16 based optimizer library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lodestone-rock/torchastic",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
