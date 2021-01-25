## General Info
This repository includes code for training a Convolutional Neural Network (CNN) to 15 different types of interferences.

Dataset Used: http://www.crawdad.org/owl/interference/20190212/index.html

We use very simple CNN model and achieve best accuracy of ~77% with 25 Epochs.

This implementation uses pytorch library and pytorch lightning trainer

## Requirements
Python3, Pytorch, Pytorch Lightning

## Setup
To run this project install the requirements and make sure you have at least 14GB GPU:
```
bash train.sh
```

## Todo
1. Use command line arguments to change training parameters e.g batch size and epochs
2. Model can be improved [loss is still converging], use more complex model and increase epochs. Current model is very simple.
3. Add comments