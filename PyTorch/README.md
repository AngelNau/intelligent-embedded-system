# PyTorch
This directory contains all the necessary files for training and quantizing MobileNetV2 and TinyCNN models using PyTorch. It includes the following files:

1. [mobileNetV2.py](mobileNetV2.py) - Contains the code for initializing, training, quantizing and exporting a MobileNetV2 model.

2. [tinyCnn.py](tinyCnn.py) - Contains the code for initializing, training, quantizing and exporting a TinyCNN model.

3. [train.py](train.py) - Contains the code used for training the models.

4. [utils.py](utils.py) - Contains the code for splitting the dataset into training, testing and validation subsets.

5. [weights.py](weights.py) - Contains the code used to transfer the weights of a pre-trained MobileNetV2 model with `width = 1` to a MobileNetV2 model with `width = 0.35`

6. [quant.py](quant.py) - Contains code for quantizing the models using the `onnx` and `onnxruntime` libraries.

3. [requirements.txt](requirements.txt) - Contains all the necessary packages required for the Python scripts along with their versions.

# Getting started
Assuming that a virtual environment already exists and is activated, the code can be run using:
```bash
python mobileNetV2.py
```
```bash
python tinyCnn.py
```
```bash
python quant.py <model_name>
```