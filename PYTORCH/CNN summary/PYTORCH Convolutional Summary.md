
# Neural networks with PyTorch

Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from. You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement. PyTorch has a nice module `nn` that provides a nice way to efficiently build large neural networks.

- EACH MNIST IMAGE IS 28*28 PIXELS
- 1 COLOR CHANNEL
- BATCHES OF 64   
`torch.Size([64, 1, 28, 28])`
- TARGET 784 INPUT 256 HIDDEN 10 OUTPUT   
  
  
**INPUT NODE** Must be 1d vector. So flatten 28*28 = `inputs = images.view(images.shape[0], -1)`


```python
# Import necessary packages

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt
```


Now we're going to build a larger network that can solve a (formerly) difficult problem, identifying text in an image. Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Each image is 28x28 pixels, you can see a sample below

<img src='assets/mnist.png'>

Our goal is to build a neural network that can take one of these images and predict the digit in the image.

First up, we need to get our dataset. This is provided through the `torchvision` package. The code below will download the MNIST dataset, then create training and test datasets for us. Don't worry too much about the details here, you'll learn more about this later.


```python
### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

    0it [00:00, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz


    9920512it [00:04, 2283063.15it/s]                             


    Extracting /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz


      0%|          | 0/28881 [00:00<?, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz


    32768it [00:00, 142174.65it/s]           
      0%|          | 0/1648877 [00:00<?, ?it/s]

    Extracting /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz


    1654784it [00:01, 949569.85it/s]                              
    0it [00:00, ?it/s]

    Extracting /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    8192it [00:00, 54786.15it/s]            

    Extracting /Users/adammcmurchie/.pytorch/MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!


    


We have the training data loaded into `trainloader` and we make that an iterator with `iter(trainloader)`. Later, we'll use this to loop through the dataset for training, like

```python
for image, label in trainloader:
    ## do things with images and labels
```

You'll notice I created the `trainloader` with a batch size of 64, and `shuffle=True`. The batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a *batch*. And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again. But here I'm just grabbing the first batch so we can check out the data. We can see below that `images` is just a tensor with size `(64, 1, 28, 28)`. So, 64 images per batch, 1 color channel, and 28x28 images.


```python
dataiter = iter(trainloader)
images, labels = dataiter.next()
print("image type is : " )
print(str(type(images)))
print('')
print("image shape is : ")
print(str(images.shape))
print('')
print('labels shape is : ')
print(str(labels.shape))
print('')

print("image labels: ")
print(labels)
```

    image type is : 
    <class 'torch.Tensor'>
    
    image shape is : 
    torch.Size([64, 1, 28, 28])
    
    labels shape is : 
    torch.Size([64])
    
    image labels: 
    tensor([8, 0, 4, 9, 2, 9, 5, 9, 8, 9, 6, 6, 1, 1, 8, 9, 8, 8, 0, 7, 7, 0, 4, 3,
            8, 4, 0, 3, 9, 0, 1, 1, 4, 0, 5, 0, 2, 0, 9, 7, 0, 9, 8, 0, 0, 3, 1, 2,
            3, 3, 4, 1, 1, 3, 3, 7, 5, 2, 4, 3, 4, 5, 1, 6])


This is what one of the images looks like. 


```python
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');
```


![png](output_8_0.png)


First, let's try to build a simple network for this dataset using weight matrices and matrix multiplications. Then, we'll see how to do it using PyTorch's `nn` module which provides a much more convenient and powerful method for defining network architectures.

The networks you've seen so far are called *fully-connected* or *dense* networks. Each unit in one layer is connected to each unit in the next layer. In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples). However, our images are 28x28 2D tensors, so we need to convert them into 1D vectors. Thinking about sizes, we need to convert the batch of images with shape `(64, 1, 28, 28)` to a have a shape of `(64, 784)`, 784 is 28 times 28. This is typically called *flattening*, we flattened the 2D images into 1D vectors.

Previously you built a network with one output unit. Here we need 10 output units, one for each digit. We want our network to predict the digit shown in an image, so what we'll do is calculate probabilities that the image is of any one digit or class. This ends up being a discrete probability distribution over the classes (digits) that tells us the most likely class for the image. That means we need 10 output units for the 10 classes (digits). We'll see how to convert the network output into a probability distribution next.

> **Exercise:** Flatten the batch of images `images`. Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units using random tensors for the weights and biases. For now, use a sigmoid activation for the hidden layer. Leave the output layer without an activation, we'll add one that gives us a probability distribution next.


```python
## Solution
def activation(x):
    return 1/(1+torch.exp(-x))

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
print("input shape : " + str(inputs.shape) + " Number of samples(rows), pixel size(columns)")
print("output shape : " + str(out.shape)) 
print('Example batch 1 : ')
print(out[0])
```

    input shape : torch.Size([64, 784]) Number of samples(rows), pixel size(columns)
    output shape : torch.Size([64, 10])
    Example batch 1 : 
    tensor([-12.7073,  -1.1568,   0.3403,  10.9176,   5.9855,   7.5564,   3.6409,
            -19.8710,   3.5251, -20.9448])


Now we have 10 outputs for our network. We want to pass in an image to our network and get out a probability distribution over the classes that tells us the likely class(es) the image belongs to. Something that looks like this:
<img src='assets/image_distribution.png' width=500px>

Here we see that the probability for each class is roughly the same. This is representing an untrained network, it hasn't seen any data yet so it just returns a uniform distribution with equal probabilities for each class.

To calculate this probability distribution, we often use the [**softmax** function](https://en.wikipedia.org/wiki/Softmax_function). Mathematically this looks like

$$
\Large \sigma(x_i) = \cfrac{e^{x_i}}{\sum_k^K{e^{x_k}}}
$$

What this does is squish each input $x_i$ between 0 and 1 and normalizes the values to give you a proper probability distribution where the probabilites sum up to one.

> **Exercise:** Implement a function `softmax` that performs the softmax calculation and returns probability distributions for each example in the batch. Note that you'll need to pay attention to the shapes when doing this. If you have a tensor `a` with shape `(64, 10)` and a tensor `b` with shape `(64,)`, doing `a/b` will give you an error because PyTorch will try to do the division across the columns (called broadcasting) but you'll get a size mismatch. The way to think about this is for each of the 64 examples, you only want to divide by one value, the sum in the denominator. So you need `b` to have a shape of `(64, 1)`. This way PyTorch will divide the 10 values in each row of `a` by the one value in each row of `b`. Pay attention to how you take the sum as well. You'll need to define the `dim` keyword in `torch.sum`. Setting `dim=0` takes the sum across the rows while `dim=1` takes the sum across the columns.

# SOFTMAX EXPLAINED

OUTPUT IS [64,10] 64 BATCHES OF 10 Number probabilities

RUNS FOR EACH COLUMN OF EVERY ROW 

e = 2.71~

Assume 1 row only (instead of 64)  

`x1 = e^x1/(e^x1 + e^x1....+ e^x10)`  
repeat up to x10  
`x10 = e^x10/(e^x1 + e^x1....+ e^x10)`

x1 + x2...x10 = 1


```python
## Solution
# torch.exp(x) for every row, give back 10 elements powered of e

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1) #transpose

probabilities = softmax(out)






"""
                             ONLY PRINTLINE STUFF
 
"""





# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
print('')

print('THIS IS OBJECTIVE FOR EACH OF 64 ELEMENTS')
print('')
print('print first probability row of 64: ' )
print(str(probabilities[0]))

print('')
# Does it sum to 1?
print('summ up each of the 64 elements')
print(probabilities.sum(dim=1))

print(' ')
print('shape of output : ' + str(out.shape))

print("Print out each elemement of bottom equation")
#print((torch.sum(torch.exp(out), dim=1)).shape)
#print(torch.sum(torch.exp(out), dim=1).view(-1, 1).shape)
print(torch.sum(torch.exp(out), dim=1).view(-1, 1))

```

    torch.Size([64, 10])
    
    THIS IS OBJECTIVE FOR EACH OF 64 ELEMENTS
    
    print first probability row of 64: 
    tensor([5.2653e-11, 5.4673e-06, 2.4432e-05, 9.5855e-01, 6.9124e-03, 3.3255e-02,
            6.6281e-04, 4.0764e-14, 5.9029e-04, 1.3930e-14])
    
    summ up each of the 64 elements
    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000])
     
    shape of output : torch.Size([64, 10])
    Print out each elemement of bottom equation
    tensor([[5.7524e+04],
            [1.1189e+03],
            [1.2192e+04],
            [8.5388e+05],
            [6.5658e+03],
            [1.3670e+03],
            [4.3213e+04],
            [3.1813e+02],
            [3.3716e+04],
            [1.1511e+04],
            [5.3802e+05],
            [1.5142e+03],
            [6.2863e+03],
            [5.0698e+03],
            [6.6175e+04],
            [6.8256e+02],
            [4.3614e+03],
            [6.2630e+05],
            [9.4566e+05],
            [2.1035e+02],
            [2.4725e+02],
            [5.5656e+03],
            [3.5514e+04],
            [5.9846e+04],
            [6.6478e+02],
            [1.5383e+03],
            [2.1019e+05],
            [2.8097e+06],
            [3.7195e+02],
            [2.3352e+04],
            [2.7334e+02],
            [1.5078e+03],
            [1.1651e+09],
            [4.6284e+04],
            [1.0940e+04],
            [1.4764e+06],
            [1.0717e+04],
            [1.0732e+05],
            [5.0166e+04],
            [3.4300e+04],
            [1.5027e+06],
            [1.6931e+03],
            [6.0747e+05],
            [2.4060e+08],
            [2.1773e+07],
            [2.4435e+04],
            [4.8679e+03],
            [2.8386e+03],
            [1.1489e+01],
            [6.3507e+02],
            [1.2034e+07],
            [2.9154e+03],
            [1.8411e+03],
            [2.0815e+04],
            [1.4049e+04],
            [7.6694e+04],
            [2.8334e+02],
            [2.2966e+02],
            [6.8185e+01],
            [2.1644e+02],
            [1.5531e+05],
            [9.0948e+03],
            [5.3013e+03],
            [1.1056e+02]])


## THIS TIME WITH NN MODULE

PyTorch provides a module `nn` that makes building networks much simpler. Here I'll show you how to build the same one as above with 784 inputs, 256 hidden units, 10 output units and a softmax output.


```python
from torch import nn
```


```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
```

Let's go through this bit by bit.

```python
class Network(nn.Module):
```

Here we're inheriting from `nn.Module`. Combined with `super().__init__()` this creates a class that tracks the architecture and provides a lot of useful methods and attributes. It is mandatory to inherit from `nn.Module` when you're creating a class for your network. The name of the class itself can be anything.

```python
self.hidden = nn.Linear(784, 256)
```

This line creates a module for a linear transformation, $x\mathbf{W} + b$, with 784 inputs and 256 outputs and assigns it to `self.hidden`. The module automatically creates the weight and bias tensors which we'll use in the `forward` method. You can access the weight and bias tensors once the network (`net`) is created with `net.hidden.weight` and `net.hidden.bias`.

```python
self.output = nn.Linear(256, 10)
```

Similarly, this creates another linear transformation with 256 inputs and 10 outputs.

```python
self.sigmoid = nn.Sigmoid()
self.softmax = nn.Softmax(dim=1)
```

Here I defined operations for the sigmoid activation and softmax output. Setting `dim=1` in `nn.Softmax(dim=1)` calculates softmax across the columns.

```python
def forward(self, x):
```

PyTorch networks created with `nn.Module` must have a `forward` method defined. It takes in a tensor `x` and passes it through the operations you defined in the `__init__` method.

```python
x = self.hidden(x)
x = self.sigmoid(x)
x = self.output(x)
x = self.softmax(x)
```

Here the input tensor `x` is passed through each operation a reassigned to `x`. We can see that the input tensor goes through the hidden layer, then a sigmoid function, then the output layer, and finally the softmax function. It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build. The order in which you define things in the `__init__` method doesn't matter, but you'll need to sequence the operations correctly in the `forward` method.

Now we can create a `Network` object.


```python
# Create the network and look at it's text representation
model = Network()
print("hidden layer shape : " + str(model.hidden.weight.shape)) # for some reason back to front
print("output shape: " + str(model.output))
```

    hidden layer shape : torch.Size([256, 784])
    output shape: Linear(in_features=256, out_features=10, bias=True)


You can define the network somewhat more concisely and clearly using the `torch.nn.functional` module. This is the most common way you'll see networks defined as many operations are simple element-wise functions. We normally import this module as `F`, `import torch.nn.functional as F`.


```python
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
```

### Activation functions

So far we've only been looking at the softmax activation, but in general any function can be used as an activation function. The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. Here are a few more examples of common activation functions: Tanh (hyperbolic tangent), and ReLU (rectified linear unit).

<img src="assets/activation.png" width=700px>

In practice, the ReLU function is used almost exclusively as the activation function for hidden layers.

### Your Turn to Build a Network

<img src="assets/mlp_mnist.png" width=600px>

> **Exercise:** Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. You can use a ReLU activation with the `nn.ReLU` module or `F.relu` function.

It's good practice to name your layers by their type of network, for instance 'fc' to represent a fully-connected layer. As you code your solution, use `fc1`, `fc2`, and `fc3` as your layer names.


```python
## Solution

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x

model = Network()
model
```




    Network(
      (fc1): Linear(in_features=784, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=64, bias=True)
      (fc3): Linear(in_features=64, out_features=10, bias=True)
    )



### Initializing weights and biases

The weights and such are automatically initialized for you, but it's possible to customize how they are initialized. The weights and biases are tensors attached to the layer you defined, you can get them with `model.fc1.weight` for instance.


```python
print(model.fc1.weight.shape) # for some reason back to front
print(model.fc1.bias.shape)
```

    torch.Size([128, 784])
    torch.Size([128])


For custom initialization, we want to modify these tensors in place. These are actually autograd *Variables*, so we need to get back the actual tensors with `model.fc1.weight.data`. Once we have the tensors, we can fill them with zeros (for biases) or random normal values.


```python
# Set biases to all zeros
model.fc1.bias.data.fill_(0)
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])




```python
# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)
```




    tensor([[ 0.0088, -0.0021,  0.0086,  ..., -0.0071, -0.0055,  0.0024],
            [-0.0023,  0.0093, -0.0040,  ...,  0.0002,  0.0092,  0.0076],
            [ 0.0055, -0.0049,  0.0045,  ...,  0.0071,  0.0032,  0.0068],
            ...,
            [-0.0033, -0.0036, -0.0030,  ..., -0.0130, -0.0011,  0.0129],
            [ 0.0027,  0.0193, -0.0025,  ..., -0.0126, -0.0162,  0.0061],
            [-0.0175, -0.0117,  0.0027,  ...,  0.0039, -0.0141, -0.0006]])



### Forward pass

Now that we have a network, let's see what happens when we pass in an image.


```python
# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
```


![png](output_30_0.png)


As you can see above, our network has basically no idea what this digit is. It's because we haven't trained it yet, all the weights are random!

### Using `nn.Sequential`

PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)). Using this to build the equivalent network:


```python
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
```

    Sequential(
      (0): Linear(in_features=784, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=10, bias=True)
      (5): Softmax()
    )



![png](output_32_1.png)


The operations are availble by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use `model[0]`.


```python
print(model[0])
model[0].weight
```

    Linear(in_features=784, out_features=128, bias=True)





    Parameter containing:
    tensor([[ 0.0099, -0.0071,  0.0277,  ...,  0.0320,  0.0015, -0.0128],
            [ 0.0115, -0.0156,  0.0294,  ..., -0.0020, -0.0031,  0.0215],
            [ 0.0189, -0.0221,  0.0297,  ...,  0.0068,  0.0158,  0.0166],
            ...,
            [ 0.0317,  0.0237, -0.0194,  ...,  0.0308, -0.0023,  0.0226],
            [-0.0208, -0.0103,  0.0334,  ..., -0.0259,  0.0156,  0.0194],
            [-0.0058, -0.0153, -0.0162,  ..., -0.0088,  0.0175,  0.0288]],
           requires_grad=True)



You can also pass in an `OrderedDict` to name the individual layers and operations, instead of using incremental integers. Note that dictionary keys must be unique, so _each operation must have a different name_.


```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model
```




    Sequential(
      (fc1): Linear(in_features=784, out_features=128, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=128, out_features=64, bias=True)
      (relu2): ReLU()
      (output): Linear(in_features=64, out_features=10, bias=True)
      (softmax): Softmax()
    )



Now you can access layers either by integer or the name


```python
print(model[0])
print(model.fc1)
```

    Linear(in_features=784, out_features=128, bias=True)
    Linear(in_features=784, out_features=128, bias=True)


In the next notebook, we'll see how we can train a neural network to accuractly predict the numbers appearing in the MNIST images.
