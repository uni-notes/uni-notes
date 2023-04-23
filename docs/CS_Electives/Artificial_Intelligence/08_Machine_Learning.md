## Machine Learning

For more insight, refer to [Machine Learning](../Machine_Learning/01_Intro.md)

### Neurons and Neural Networks

Kindly refer [Artificial Neural Networks](../Machine_Learning/12_ANN.md)


### RNN

A recurrent neural network (RNN) is a kind of artificial neural network mainly used in speech recognition and natural language processing (NLP).  

A recurrent neural network looks similar to a traditional neural network except that a memory-state is added to the neurons.

![RNN diagram](assets/rnn.png)

### CNN

Convolutional Neural Networks is a type of architecture that exploits special properties of image data and are used in computer vision applications.

Images are a 3-dimensional array of features: each pixel in the 2-D space contains three numbers from 0–255 (inclusive) corresponding to the Red, Green and Blue.

The first important type of layer that a CNN has is called the Convolution (Conv) layer. It uses parameter sharing and applies the same smaller set of parameters spatially across the image. 

Essentially the parameters (i.e. weights) associated to the input remain the same but the input itself is different as the layer computes the output of the neurons at different regions of the image.

Hyperparameters of conv layers are 
- Filter size - corresponds to how many input features in the width and height dimensions one neuron takes in
- Stride - how many pixels we want to move (towards the right/down direction) when we apply the neuron again

Then we have the pooling layer. The purpose of the pooling layer is to reduce the spatial size (width and height) of the layers. This reduces the number of parameters (and thus computation) required in future layers.

We use a fully connected layers at the end of our CNNs. When we reach this stage, we can flatten the neurons into a one-dimensional array of features.

![CNN diagram](assets/cnn.png)

### LSTM

> Notes for LSTM need to be added