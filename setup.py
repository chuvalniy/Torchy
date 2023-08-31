from setuptools import setup, find_packages

VERSION = '0.2.3.2'
DESCRIPTION = 'NumPy based neural network package'

setup(
    name="torchy-nn",
    version=VERSION,
    author="Valentin Belyaev",
    author_email="chuvalik.work@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/chuvalniy/Torchy",
    install_requires=['numpy', 'scikit-learn', 'pytest'],
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
