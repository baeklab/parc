# PARC: Physics-Aware Recurrent Convolutions

## Getting Started

### Virtual Environment
It is recommended to create a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install PARC within the virtual environment. To create a virtual environment, type the following command in your terminal or command prompt.
```bash
conda create -n parc python=3.8 ipykernel
```
Once the virtual environment has been created, run the following to activate the environment.
```bash
conda activate parc
```

### Dependencies

#### TensorFlow
Tested and developed on TensorFlow 2.8.0. It should be compatible with other TensorFlow2 versions, but we haven't tested. Make sure you follow the installation instructions in https://www.tensorflow.org/install to install TensorFlow2 according to your system configuration.
```bash
pip install tensorflow
```

#### Other Dependencies
If you have successfully configured TensorFlow2, installation of other dependencies can be as simple as running the following command.
```bash
pip install -r requirements.txt
```

### Clone This Repository
```bash
git clone https://github.com/stephenbaek/parc.git
```

```bash
cd parc
```

### Run Examples
