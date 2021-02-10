# Classifying-Handwritten-Digits
Classifying Handwritten MNIST Digits Using TensorFlow.

<p align="center">
  <img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/MNIST.jpeg" width="700">
</p>

# Project Objective
* The project objective is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

# What is MNIST Classification Problem ?
* The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.
* The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.
* It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

# Layers in a CNN
We are capable of using many different layers in a convolutional neural network. However, convolution, pooling, and fully connected layers are the most important ones.

## Convolutional Layers
The convolutional layer is the very first layer where we extract features from the images in our datasets. Due to the fact that pixels are only related to the adjacent and close pixels, convolution allows us to preserve the relationship between different parts of an image. 

<p align="center">
  <img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/Conv.png" width="700">
</p>

## Pooling Layer
When constructing CNNs, it is common to insert pooling layers after each convolution layer to reduce the spatial size of the representation to reduce the parameter counts which reduces the computational complexity. In addition, pooling layers also helps with the overfitting problem. 

## Fully Connected Layers
A fully connected network is our RegularNet where each parameter is linked to one another to determine the true relation and effect of each parameter on the labels. 

# Mechanism
<p align="center">
  <img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/MNIST.gif" width="700">
</p>

# Libraries Used
* TensorFlow
* Matplotlib
* Keras
* SGD Optimizer

# Loading Data
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

# Images Reshape
```python
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
```

# Model Structure
* Input_shape = (28, 28, 1)
* Convolution layer with 28 filter and kerel size (3,3).
* MaxPooling layer with size (2,2).
* Fully connected layer with 128 neurons with activation function ReLU.
* Dropout layer with 0.2.
* Fully connected layer with 10 neurons with activation function ReLU.
* EPOCHS = 10
* Optimizer during complile is Adam.
* Loss during compile is Categorical Crossentropy

# Testing
```python
image_index = 9431
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
```

# Results
> Model Accuracy is 98.44%.

<img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/Result1.PNG" width="250" height="250"><img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/Result2.PNG" width="250" height="250"><img src="https://github.com/OmarReda/Classifying-Handwritten-Digits/blob/main/Result3.PNG" width="250" height="250">

