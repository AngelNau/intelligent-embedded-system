# Intelligent Embedded System

This project contains code used in my diploma thesis for  initializing, training, quantizing and exporting 2 AI models, MobileNetV2 and my own TinyCNN.

The [stm32ai-modelzoo-services](stm32ai-modelzoo-services/) directory is a submodule from a fork of the stm32ai-modelzoo-services [repository](https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git). It is slightly changed so that it works with the structure of this project.

##  Getting Started

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/AngelNau/intelligent-embedded-system.git
```
### 2. Creating virtual environments
To prevent package version mismatches, we recommend creating separate Python virtual environments for [PyTorch](PyTorch/), [TensorFlow](TensorFlow/) and [stm32ai-modelzoo-services](stm32ai-modelzoo-services/).

Before creating the virtual environment, make sure to check wether `Python3.11` and `Python3.10` are installed on your system:
```bash
python3.10 --version
python3.11 --version
```
It should print something like `Python 3.11.x/Python 3.10.x`. If not, download them [here](https://www.python.org/downloads/).\
`Python3.10` is used for [stm32ai-modelzoo-services](stm32ai-modelzoo-services/), `Python3.11` is for [PyTorch](PyTorch/) and [TensorFlow](TensorFlow/).

Make sure to cd into each directly seperately, each of them use their own separate packages:
* **Linux/MacOS**
```bash
cd PyTorch/
```

#### Option A: Using `venv` (built-in)
1. Create a new virtual environment in a `.venv` folder:
	```bash
	python3.11 -m venv .venv
	```
	
2. Activate the environment:
	* **Linux/MacOS**
	```bash
	source .venv/bin/activate
	```

3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

#### Option B: Using `uv` (faster package manager)

[`uv`](https://github.com/astral-sh/uv) is a modern drop-in replacement for `pip` + `venv`.
It creates environments and installs dependencies **much faster**.

1. Create a `.venv` folder with Python 3.11:
	```bash
	uv venv --python 3.11
	```

2. Activate the environment:
	* **Linux/MacOS**
	```bash
	source .venv/bin/activate
	```

3. Install dependencies:
	```bash
	uv pip install -r requirements.txt
	```

### 3. Running the code
Once the virtual environment has been created, activated, and all the dependencies have been installed, simply run the code using:
```bash
python <filename.py>
```

### 4. Disabling the environment
Once finished with running the Python scripts, you should disable the Python virtual environment. To do this run the following command:
```bash
deactivate
```

## Additional documentation
Inside each directory, there is an additional README.md file explaining the project further, please refer to them once the you've setup the virtual environments.\
[PyTorch README.md](PyTorch/README.md)\
[TensorFlow README.md](TensorFlow/README.md)\
[stm32ai-modelzoo-services README.md](stm32ai-modelzoo-services/README-1.md)