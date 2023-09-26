
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI Version](https://img.shields.io/pypi/v/torchy-nn.svg)](https://pypi.org/project/torchy-nn/)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)

![изображение](https://github.com/chuvalniy/Torchy/assets/85331232/e0ab8cfe-4e12-42f9-b90e-37fb93f8ffd0)


## Overview
Torchy is a neural network framework implemented only using NumPy and based on PyTorch API but with manual backpropogation calculations. The main idea was to build a neural network from scratch for educational purposes.

## Installation
```python
pip install torchy-nn
```
## Getting started
I suggest you to take a look at [currently implemented stuff](https://github.com/chuvalniy/Torchy/blob/main/docs/Implemented.md) to be familiar with current possibilities for building neural network models with Torchy. Also I've created [package structure](https://github.com/chuvalniy/Torchy/blob/main/docs/PackageStructure.md) in case if you stuck where to get specific layers.

## Example usage
First we can define our model using Torchy with its PyTorch-like API

```python
import torchy.nn as nn  # Same as torch.nn

# Define 2-layer wtth 100 neurons hidden layer.
model = nn.Sequential(
    nn.Linear(n_input=10, n_output=100),
    nn.BatchNorm1d(n_output=100),
    nn.ReLU(),
    nn.Linear(n_input=100, n_output=2)
)
```

Next step is to create instances of optimizer and criterion for loss function and scheduler for fun

```python
import torchy.optimizers as optim

optimizer = optim.SGD(model.params(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.StepLR(optimizer, step_size=10)
```

I won't cover whole training process like loops and stuff, just show you main differences while training

```python
...
predictions = model(X)  # Nothing changed

loss, grad = criterion(predictions, y)  # Now return tuple of (loss, grad) instead of only loss 

optimizer.zero_grad()
model.backward(grad)  # Call backward on model object and pass gradient from loss as argument
optimizer.forward_step()
```

## Testing
Every neural network module is provided with unit test that verify correctness of forward and backward passes.

Execute the following command in your project directory to run the tests.
```python
pytest -v
```
## Demonstration
The [demo notebook](https://github.com/chuvalniy/Torchy/blob/main/torchy-demo.ipynb) showcases what Torchy currently can do.

## Roadmap
It seems to me that I've completed all the modules that I wanted to code, but there are still some unfinished points, here are some of them:
- Dataset, DataLoader & Sampler;
- Transformer;
- Fix regularization;
- Autograd (as new git branch)

## Resources
The opportunity to create such a project was given to me thanks to these people

- [PyTorch](https://github.com/pytorch/pytorch)
- [CS231n - 2016](https://youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
- [Deep Learning на пальцах - 2019](https://youtube.com/playlist?list=PL5FkQ0AF9O_o2Eb5Qn8pwCDg7TniyV1Wb)
- [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)


## License
[MIT License](https://github.com/chuvalniy/Torchy/blob/main/LICENSE)

