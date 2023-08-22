# Neural networks

Referred to also as deep learning.
- speech recognition
- computer vision/ images
- text

Neurons take in numbers and output other numbers after doing some computation

a= f(x) from logistic regression -> stands for activation
thought of as a simplified model of a neuron in the brain
Input = x output = a = probability of something occurring
logistic regression model is one neutron


We create artifical neurons that might be combinations of different features and present interpretable things like affordability (price, shipping cost). If we also create aritifical neurons for awareness and perceived quality from different combinations of original features we create three neurons and one artifical layer.
The number outputs from the artifical neurons are called activations
- output layer contains final probability
- which information does an aritifical neuron take from the previous layer -> each neuron has access to every feature from the previous layer

input feature vector = x
activation value vector = a
output layer output = probability - final unit uses logistic regression but with activations as features - so the model learns its own features

artificial neuron layer = hidden layer -> can have multiple hidden layers
how many neurons to use
how many hidden layers
this is the neuron architecture
also called a multi layer perceptron

Face recognition:
Pixels = 1000 -> 1000x1000 matrix of pixel intensity values -> put these into a single vector of 1 million length (1000 ^ 2)

Earliest neurons are looking for short lines
Second layer neurons learn to group together lines
Next hidden layer neuron is aggregating different parts of faces

Equation of the activation function of a layer:
a= activation
l = layer
j = Jth unit of the layer
g = sigmoid function


aj^[l] = g(wj^[l] * a^[l-1] + bj^[l])