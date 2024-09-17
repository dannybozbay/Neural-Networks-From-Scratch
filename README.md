<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<h1 align="center"> Neural Networks From Scratch!</h1> 

<div align="center">
    <img src="https://images.theconversation.com/files/374303/original/file-20201210-18-elk4m.jpg?ixlib=rb-4.1.0&rect=0%2C22%2C7500%2C5591&q=45&auto=format&w=926&fit=clip" alt="Logo" width="500" height="300">
  </a>

</div>

## About The Project

Curious about how deep learning libraries work? Neural Networks From Scratch is a hands-on project that walks you through building neural networks from the ground up. It includes Python scripts covering core concepts like network architectures, training algorithms, and performance metrics.

Additionally, there's an educational [Kaggle notebook](https://www.kaggle.com/code/danielbozbay/building-a-neural-network-from-scratch-numpy) that breaks down the framework step by step!

## What It Does

* **Builds Neural Networks**: Implements a series of improved neural network frameworks with different architectures, cost functions, and optimization techniques.
* **Trains Models**: Uses stochastic gradient descent (SGD) and backpropagation to train networks on real datasets like MNIST.
* **Evaluates Performance**: Monitors and plots training and validation metrics to assess model performance.
* **Compares Techniques**: Evaluates different cost functions, regularization strategies, and initialization methods to highlight their impact on network performance.

## Who It's For

This project is aimed at students, researchers, and practitioners interested in gaining a deeper understanding of neural network mechanics and implementation. It's particularly useful for those who want to:

* Learn the fundamentals of neural network design and training without relying on high-level libraries.
* Experiment with different network configurations and optimization techniques.
* Gain hands-on experience with practical machine learning tasks and performance evaluation.

Whether you're new to neural networks or looking to strengthen your understanding, "Neural Networks From Scratch" provides the tools and insights to explore and experiment with fundamental concepts in machine learning.



## Installation

To get started with the Neural Networks From Scratch project, follow these steps to set up your development environment:

**1. Clone the Repository**

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/dannybozbay/Neural-Networks-From-Scratch
```

**2. Navigate to the Project Directory**

Change to the project directory:

```bash
cd Neural-Networks-From-Scratch
```

**3. Create a Python Virtual Environment**

It's recommended to use a virtual environment to manage dependencies. Create a virtual environment with venv:

```bash
python3 -m venv venv
```

Activate the virtual environment:

* On **Windows**:

    ```bash
    venv\Scripts\activate
    ```

* On **macOS/Linux**:

    ```bash
   source venv/bin/activate
    ```

**4. Install Dependencies**

Install the required packages listed in *requirements.txt*:

```bash
pip install -r requirements.txt
```

**5. Start Using the Project**

You are now ready to use the project. Follow the documentation and examples provided in the repository to get started.

For any issues or questions, refer to the Issues page or contact the project maintainers.
## Data

This project uses the MNIST dataset for testing and evaluating the neural network models. MNIST (Modified National Institute of Standards and Technology) is a widely used benchmark dataset in machine learning and computer vision, consisting of handwritten digits.


### MNIST Dataset Overview

* **Content**: The dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is a 28x28 pixel grayscale image.
* **Purpose**: MNIST is commonly used to train and evaluate image classification algorithms due to its simplicity and well-defined problem scope.

### Data Loading

The MNIST dataset is included in the repository and will be automatically available when you clone the project. Specifically:

* **File Location**: The `mnist.pkl` file is stored in the `Data` folder within the repository.
* **Automatic Download**: When you clone the repository, the `mnist.pkl` file is already included in the correct location. No additional steps are required to obtain the dataset.

The dataset is loaded using the `mnist_loader` module included in the project. This module handles downloading, preprocessing, and splitting of the MNIST data into training, validation, and test sets. The data is automatically formatted to be compatible with the neural network models provided in this project.

[![MNIST-Database.png](https://i.postimg.cc/dDPt0WbT/MNIST-Database.png)](https://postimg.cc/7J9ysNYx)

## Roadmap

This project is structured to facilitate the development, training, and evaluation of neural network models using the MNIST dataset. Below is an overview of the project's structure and the roles of its key components:


```
Neural-Networks-From-Scratch/
├── data/
│   └── mnist.pkl.gz
├── reports/
│   └── figures/
│       ├── network/
│       ├── network_matrix/
│       ├── network2/
│       └── network2_matrix/
├── scripts/
│   ├── run_avg_darkness.py
│   ├── run_network.py
│   ├── run_network_matrix.py
│   ├── run_network2_matrix.py
│   └── run_svm.py
└── src/
    ├── baselines/
    │   ├── __init__.py
    │   ├── mnist_avg_darkness.py
    │   └── mnist_svm.py
    ├── data/
    │   ├── __init__.py
    │   └── mnist_loader.py
    ├── networks/
    │   ├── __init__.py
    │   ├── network.py
    │   ├── network_matrix.py
    │   ├── network2.py
    │   └── network2_matrix.py
    └── util/
        ├── __init__.py
        └── plots.py
```

### Key Components

* **`data/`**: This directory contains the `mnist.pkl.gz `file, which stores the MNIST dataset. The dataset is used for training and evaluating the neural network models.

* **`reports/figures/`**: This directory stores the output figures generated during model training and evaluation. Each subfolder corresponds to a specific network implementation (`network`, `network_matrix`, `network2`, `network2_matrix`).

* **`scripts/`**: The scripts folder contains Python scripts that run the training and evaluation of different models:

     * `run_avg_darkness.py`: Executes the average darkness baseline model.

     * `run_svm.py`: Executes the SVM baseline model
     * `run_network.py`: Trains and evaluates the `network` model.
     * `run_network_matrix.py`: Trains and evaluates the `network_matrix` model for improved training speed.

    * `run_network2_matrix.py`: Trains and evaluates the advanced `network2_matrix` model for improved performance and training speed.

* **`src/`**: The core of the project, containing the main modules:

    * `baselines/`: Contains simple baseline models:

        * `mnist_avg_darkness.py`: Implements a basic model using average darkness.
        * `mnist_svm.py`: Implements a Support Vector Machine (SVM) model.

    * `data/`: Responsible for loading and preprocessing the MNIST data:
        * `mnist_loader.py`: A module that loads and preprocesses the MNIST dataset for use in training and evaluation.

    * `networks/`: Contains neural network implementations:
        * `network.py`: A simple, initial implementation of a neural network.
        * `network_matrix.py`: An optimized version of network using matrix operations for improved training speed.
        *   `network2.py`: A more advanced network with features like different cost functions, L2 regularization, and improved weight initialization.
        * `network2_matrix.py`: An optimized version of `network2` using matrix operations for improved training speed.

    * `util/`: Contains utility functions:
        * `plots.py`: Functions and settings for generating and saving plots.

## Usage/Examples

Below is a simple example using the `run_network.py `script to demonstrate the workflow of loading data, initializing a neural network, training it, and visualizing the results. This script specifically handles the MNIST dataset, initializes a neural network with a single hidden layer, trains the network using stochastic gradient descent (SGD), and plots the accuracy over training epochs. 

```python
from data import mnist_loader
from networks import network
from util.plots import plot_metrics

# Step 1: Load and preprocess the MNIST dataset
train, validation, test = mnist_loader.load_data_wrapper()

# Step 2: Initialize the neural network
# The network has 784 input neurons (for the 28x28 pixel images), 
# one hidden layer with 30 neurons, and 10 output neurons (for the 10 digit classes)
net = network.Network([784, 30, 10])

# Step 3: Train the network using stochastic gradient descent (SGD)
# Training for 30 epochs with mini-batches of size 10 and a learning rate of 3.0
# Also, track training and validation accuracy for monitoring
training_accuracy, validation_accuracy = net.SGD(
    training_data=train,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    validation_data=validation,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)

# Step 4: Plot the accuracies over the epochs
accuracy_plot = plot_metrics(
    [training_accuracy, validation_accuracy],
    ["Training Data", "Validation Data"],
    is_accuracy=True,
)

# Step 5: Save the accuracy plot to a file
accuracy_plot.savefig(
    "../reports/figures/network/accuracy_layers_784_30_10_eta_3_epochs_30.png"
)


```


## Screenshots

[![final-plot.png](https://i.postimg.cc/c1DRJfkk/final-plot.png)](https://postimg.cc/HjycZ8gb)

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Acknowledgements

 - [Neural Networks and Deep Learning - Michael Neilson](http://neuralnetworksanddeeplearning.com/)

[contributors-shield]: https://img.shields.io/github/contributors/dannybozbay/Neural-Networks-From-Scratch.svg?style=for-the-badge
[contributors-url]: https://github.com/dannybozbay/Neural-Networks-From-Scratch/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/dannybozbay/Neural-Networks-From-Scratch.svg?style=for-the-badge
[forks-url]: https://github.com/dannybozbay/Neural-Networks-From-Scratch/ƒƒƒƒnetwork/members

[stars-shield]: https://img.shields.io/github/stars/dannybozbay/Neural-Networks-From-Scratch.svg?style=for-the-badge
[stars-url]: https://github.com/dannybozbay/Neural-Networks-From-Scratch/stargazers

[issues-shield]: https://img.shields.io/github/issues/dannybozbay/Neural-Networks-From-Scratch.svg?style=for-the-badge
[issues-url]: https://github.com/dannybozbay/Neural-Networks-From-Scratch/issues

[license-shield]: https://img.shields.io/github/license/dannybozbay/neural-networks-from-scratch.svg?style=for-the-badge
[license-url]: https://github.com/dannybozbay/neural-networks-from-scratch/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dannybozbay

