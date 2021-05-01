# CS4933_Project | Colorizing GANs


In this work, we generalize the colorization operation using a (DCGAN) with a U-Net generator and PatchGAN descriminator. This network is trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

## Prerequisites
* Linux
* Tensorflow 1.7
* Nvidia K80 / T4 (12GB / 16GB) 2 Cores

# Getting Started 
* Be sure to attach your Google Drive via the following:
```
from google.colab import drive
drive.mount('/content/drive')
```
* Install Scipy 1.1.0 to allow scipy.misc to functiona correctly
# Training 
* To train the model, please update the config dictionary mode value 
* Mode 0 = Train 
```python
options = {
    "seed": 100,
    "beta1": 0.0,
    "name": name.upper(),
    "mode": 0,
```
# Test
* To train the model, please update the config dictionary mode value 
* Mode 1 = Test
```python
options = {
    "seed": 100,
    "beta1": 0.0,
    "name": name.upper(),
    "mode": 1,
```
* As well as update the test_input and test_output keys
```python
    "test_input": './checkpoints/test', # test image(s) path
    "test_output": '/checkpoints/output', # output image(s) path
```
# Visual Turing Test
* To evaluate the model qualitatively using visual Turing test, please update the mode key
* Mode 2 = Visual Turing
```python
options = {
    "seed": 100,
    "beta1": 0.0,
    "name": name.upper(),
    "mode": 2,
```
* To apply time-based visual Turing test run (2 seconds decision time):
```python
"turing_test_delay": 2,
```
# Networks Architecture
## U-Net
The architecture of generator is inspired by U-Net: The architecture of the model is symmetric, with n encoding units and n decoding units. The contracting path consists of 4x4 convolution layers with stride 2 for downsampling, each followed by batch normalization and Leaky-ReLU activation function with the slope of 0.2. The number of channels are doubled after each step. Each unit in the expansive path consists of a 4x4 transposed convolutional layer with stride 2 for upsampling, concatenation with the activation map of the mirroring layer in the contracting path, followed by batch normalization and ReLU activation function. The last layer of the network is a 1x1 convolution which is equivalent to cross-channel parametric pooling layer. We use tanh function for the last layer.

![U-Net](https://raw.githubusercontent.com/AronPerez/CS4933_Project/main/images_for_README/unet.png)

## PatchGAN
For discriminator, we use patch-gan architecture with contractive path similar to the baselines: a series of 4x4 convolutional layers with stride 2 with the number of channels being doubled after each downsampling. All convolution layers are followed by batch normalization, leaky ReLU activation with slope 0.2. After the last layer, a sigmoid function is applied to return probability values of 70x70 patches of the input being real or fake. We take the average of the probabilities as the network output!

![PatchGAN](https://raw.githubusercontent.com/AronPerez/CS4933_Project/main/images_for_README/Architecture-of-the-PatchGAN-Discriminator-network.png)
