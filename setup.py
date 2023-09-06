from setuptools import setup, find_packages

VERSION = '0.3.4.3'
DESCRIPTION = 'NumPy based neural network package with PyTorch-like API'

with open("README.md", "r") as fn:
    long_description = fn.read()

setup(
    name="torchy-nn",
    version=VERSION,
    author="Valentin Belyaev",
    author_email="chuvalik.work@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/chuvalniy/Torchy",
    install_requires=['numpy', 'pytest'],
    license="MIT",
    keywords=['python', 'neural net', 'from scratch', 'numpy', 'pytorch-like', 'cnn', 'dense'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
