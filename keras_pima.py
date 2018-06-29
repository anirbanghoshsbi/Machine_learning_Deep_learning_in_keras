# usage python keras_pima.py -i 100 -f pima-indians-diabetes.csv -bs 25

# creating a Multi layered Perceptron in Keras

#import the libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import argparse
from csv import reader

# create the argument parser
ap=argparse.ArgumentParser()
ap.add_argument('-i','--epochs',required =True , type=int , help = 'the number of epochs')
ap.add_argument('-f','--dataset', required =True , help = 'path to the file')
ap.add_argument('-bs','--batch', required =True , type=int, help = 'batch size')
args= vars(ap.parse_args())


np.random.seed(7)
# load pima indians dataset we can use pandas or numpy but lets do it the plain old way 
# not very efficient but good for learning.
dataset =[]
with open(args['dataset'], 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		dataset.append(row)

#convert the list to numpy array......

dataset= np.array(dataset)

#split the dataset into X input  and y output variable

X =dataset[:,:-1]
y=dataset[:,-1]
input_dims =8 # number of features in the dataset....
# create a model [8 input dims] ---->[12 hidden dims ] ------> [1 output dim: as binary classifier]
model =Sequential()
''' we have eight input features therefore the input_dims =8 (most important thing to remember in keras), 
there is one hidden layer having eight neurons and finally the output layer has 1 neuron.'''
model.add(Dense(12 , input_dim = input_dims , activation= 'relu'))
model.add(Dense(input_dims , activation='relu'))
#binary classification , hence use activation 'sigmoid'
model.add(Dense(1,activation = 'sigmoid'))

#Compile the model 
model.compile(loss = 'binary_crossentropy' , optimizer ='adam' , metrics =['accuracy'])

#fit the model
model.fit(X , y, epochs = args['epochs'], batch_size= args['batch'])

#evaluate the model
scores = model.evaluate(X , y)

print('The model {} is {}'.format(model.metrics_names[1] , scores[1]*100))

