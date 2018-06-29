# Machine_learning_Deep_learning_in_keras
Use sklearn and keras for ML and DL. Doing Machine Learning is very easy using keras and sklearn.Both the API's work seamlessly with python .
# Multi Layer Perceptron
The field of Artificial Neural Network often called Neural Network , is a field of study that investigates how simple models that mimic biological brain can be used for solving difficult computatational tasks like predictive modelling task.
# The power of the Neural Network
The real power of the neural network comes from the fact that they learn the representation between the input and the output data. f(x)-->y the model learns how the input data is related to the output data or in other words what is the 'mapping' between the input and the output thereby making it easy for us to replace x and y to a and b get an prediction f(a)--->b.

A perceptron is a single neuron and Multi layer perceptron is a layers of perceptron stacked up.The power to learn the representation (predictive capability) comes from the multilayered nature of the Multi layer perceptron.The perceptron at various levels learn different features at different scales and combine then together to form higher order features.Examples to lines to combination of lines and from lines to shape.

# Perceptron

The perceptron is the building block of the Artificial Neural Network.These simple computational units have weighted input signal and produce an output using a activation function that is non linear like sigmoid or relu or leaky relu or tanH etc.

# Neuron weights
The neuron weights are similar to the coefficients used in a regression equation. These weights are kept small and randomto prevent complexity and fragility of the model.One target in  training a Multi level Perceptron is to get the weights that map the inputs to the output, so that we can just use the weights to make furthur prediction  and is one of te reason why inference can be drawn using excel sheets if only we have our trained weights. 

# Activation
The weighted inputs are passed through an activation function , sometimes called transfer function. An activation is the mapping between the weighted inputs to the outputs of a neuron.The activation fuction is called so because it governs threshold at which the neuron  is activated and the strength of the activation. In most case these activation functions are non linear as it allows the system to learn deeper and richer features .The activation function can be said to be the mapping function that connects the inputs to the outputs.

# Networked Neurons : Multi layered Perceptron 
layers of neurons stacked over one another. A row of neuron is called layer and one network can have multiple layers.
1. Input layer :  the layer that feeds in the data.
2. Hidden layers : any layer between the input layer and the output layer.

3.Output layer : the layer that gives the out prediction.
a.Linear regression problem might have a single output neuron and the neuron may not have any activation function.
b.A binary classifier may have single output with an added sigmoid activation function that outputs value between 0 and 1 to predict the class of the primary class.
c.A multi-class classification problem may be a single output for each class and for _five classes_ I would have five outputs.We use softmax classifier to predict the probability of the various classes and then selecting the class with the highest probabilty as the prediction.
